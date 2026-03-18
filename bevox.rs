/*! # Voxel Plugin
Bevy plugin to manage a voxel world based on a sparse octree hierarchy for level-of-detail
rendering.

User-facing Primary API:
- `Volume` components represent objects or regions of space that are filled in by voxel data
- `Volume` entites are positioned in the world with a `Transform` component
- `Volume`s are generic on a `Voxel` type, allowing users to store arbitrary spatial data
    - There are some provided Voxel types for convenience.
- `Volume`s recieve a fn(Vec3) -> Voxel, used for generation
- `Volume`s recieve a fn([Voxel; 8]) -> Vec3, used for placing vertices on the dual grid
    - There are a number of provided vertex placers, including "boxy", "smooth", and "contoured"
- `Volume`s recieve a depth (u8), delimiting maximum hierarchy/subdivision depth
- Generation and meshing of volumes happen asynchronously.
- Camera entities with the `Viewer` component determine the focal points of level-of-detail.

Eventual goals for the API:
- Voxel coordinate system within a `Volume`
- Iterate through a shape of voxels of arbitrary depth within a `Volume`
- CSG operations on Volumes
- Avian physics colliders for Volumes
- Raycast against volume's physics collider to get voxel coordinates
- Spawn interior or surface features as new entities with arbitrary components
- Save/load volumes from disk
- Synchronization over the network, view-based interest management for clients on servers
- Splitting/fracturing volumes
*/

use bevy::{prelude::*, tasks::Task};
use itertools::iproduct;

fn octants<T: Copy>(min: T, max: T) -> impl Iterator<Item = (T, T, T)> {
    iproduct!([min, max], [min, max], [min, max])
}

#[derive(Component)]
pub(crate) struct GenerateTask(pub Task<Volume>);

/// # Meshing
mod meshing {
    use super::{octants, Subsection, Volume, VolumeIndex, VolumeMap, VolumeQuery, Voxel};
    use bevy::{platform::collections::HashMap, prelude::*};

    pub struct Cell {
        pub voxels: [Voxel; 8],
    }

    #[derive(Eq, Hash, PartialEq)]
    struct CellIndex {
        pub x: usize,
        pub y: usize,
        pub z: usize,
        pub scale: u8,
    }

    impl CellIndex {
        fn new(x: usize, y: usize, z: usize, scale: u8) -> Self {
            Self { x, y, z, scale }
        }
    }

    #[derive(Default)]
    pub(crate) struct Mesher {
        base: HashMap<CellIndex, Cell>,
        parents: HashMap<CellIndex, Cell>,
    }

    impl Mesher {
        pub fn from_cells(
            index: VolumeIndex,
            volume_map: VolumeMap,
            volumes: Query<&Volume>,
        ) -> Option<Self> {
            let query_for = |index: &VolumeIndex| VolumeQuery {
                coarse: true,
                index: *index,
                meshing: true,
            };

            let mut contexts = vec![];
            for index in octants(0, 1).map(|(x, y, z)| index.offset(x, y, z)) {
                if let Some(context) = volume_map.query(query_for(&index)) {
                    if let Some(parent_context) = context
                        .index
                        .parent_index()
                        .and_then(|parent_index| volume_map.query(query_for(&parent_index)))
                        .or(Some(context))
                    {
                        contexts.push((context, parent_context));
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            }

            let mut mesher = Self::default();
            for (context, parent_context) in contexts {
                let volume = volumes.get(context.entity).unwrap();
                for ([x, y, z], cell) in volume.cells(Subsection::None) {
                    mesher
                        .base
                        .insert(CellIndex::new(x, y, z, context.index.scale()), cell);
                }

                let parent_volume = volumes.get(parent_context.entity).unwrap();
                for ([x, y, z], cell) in parent_volume.cells(Subsection::None) {
                    mesher
                        .parents
                        .insert(CellIndex::new(x, y, z, parent_context.index.scale()), cell);
                }
            }

            Some(mesher)
        }
    }
}
use meshing::{Cell, Mesher};

/// # Voxels
mod voxel {
    #[derive(Clone, Copy, Default)]
    pub struct Voxel {
        material_id: u8,
        density: u8,
    }

    impl Voxel {
        pub fn new(material_id: u8, density: f32) -> Self {
            let bound = (u8::MAX / 2) as f32;
            Self {
                material_id,
                density: (density.clamp(-bound, bound) + bound) as u8,
            }
        }

        /// Rescales discrete density values into floating point values from -1 to 1.
        pub fn density(&self) -> f32 {
            let density = self.density as f32;
            let max = u8::MAX as f32;
            (density * 2. - max) / max
        }

        pub fn opaque(&self) -> bool {
            self.density > u8::MAX / 2
        }
    }
}
use voxel::Voxel;

/// # Subsections
mod subsection {
    use std::ops::Range;

    #[derive(Clone, Copy)]
    pub enum Quadrant {
        None,
        LoLo,
        LoHi,
        HiLo,
        HiHi,
    }

    #[derive(Clone, Copy)]
    pub enum Half {
        None,
        Lo,
        Hi,
    }

    #[derive(Clone, Copy)]
    pub enum Subsection {
        XFace(Quadrant, Quadrant),
        YFace(Quadrant, Quadrant),
        ZFace(Quadrant, Quadrant),
        XYEdge(Half, Half),
        XZEdge(Half, Half),
        YZEdge(Half, Half),
        XYZCorner,
        None,
    }

    impl Subsection {
        pub fn to_ranges(&self, width: usize, cap: usize) -> [Range<usize>; 3] {
            let (offset_1, offset_2, stride) = match self {
                Self::XFace(q1, q2) | Self::YFace(q1, q2) | Self::ZFace(q1, q2) => {
                    let set = |quadrant: &Quadrant| match quadrant {
                        Quadrant::LoLo => (0, 0, 2),
                        Quadrant::LoHi => (0, width / 2, 2),
                        Quadrant::HiLo => (width / 2, 0, 2),
                        Quadrant::HiHi => (width / 2, width / 2, 2),
                        Quadrant::None => (0, 0, 1),
                    };

                    let (o1, o2, s1) = set(q1);
                    let (o3, o4, s2) = set(q2);

                    (o1 + o3, o2 + o4, s1 * s2)
                }

                Self::XYEdge(h1, h2) | Self::XZEdge(h1, h2) | Self::YZEdge(h1, h2) => {
                    let set = |half: &Half| match half {
                        Half::Lo => (0, 2),
                        Half::Hi => (width / 2, 2),
                        Half::None => (0, 1),
                    };

                    let (o1, s1) = set(h1);
                    let (o2, s2) = set(h2);

                    (o1 + o2, 0, s1 * s2)
                }

                _ => (0, 0, 1),
            };

            let range = |offset| offset..(offset + width / stride - cap);

            match self {
                Self::XFace(_, _) => [0..1, range(offset_1), range(offset_2)],
                Self::YFace(_, _) => [range(offset_1), 0..1, range(offset_2)],
                Self::ZFace(_, _) => [range(offset_1), range(offset_2), 0..1],

                Self::XYEdge(_, _) => [0..1, 0..1, range(offset_1)],
                Self::XZEdge(_, _) => [0..1, range(offset_1), 0..1],
                Self::YZEdge(_, _) => [range(offset_1), 0..1, 0..1],

                Self::XYZCorner => [0..1, 0..1, 0..1],
                Self::None => [range(0), range(0), range(0)],
            }
        }
    }
}
use subsection::{Half, Quadrant, Subsection};

/// # Volumes
mod volume {
    use super::{octants, Cell, Subsection, VolumeIndex, Voxel};
    use crate::Viewer;
    use bevy::{
        prelude::*,
        tasks::{AsyncComputeTaskPool, Task},
    };
    use itertools::iproduct;

    const GRID_SIZE: usize = 32;

    /// A cubical grid of voxels, also known as a "chunk". Will automatically be subdivided (octree-
    /// style) by [`VoxelsPlugin`] systems.
    #[derive(Clone, Component)]
    #[require(GlobalTransform)]
    pub struct Volume {
        voxels: [Voxel; GRID_SIZE * GRID_SIZE * GRID_SIZE],
        sampler: fn(Vec3) -> Voxel,
        depth: u8,
        empty: bool,
        index: VolumeIndex,
    }

    impl Volume {
        /// Generate a `GridVolume`. Needs a 3D sampling function, the transform for this `GridVolume`,
        /// and a maximum subdivision depth.
        pub(crate) fn generate(
            sampler: fn(Vec3) -> Voxel,
            transform: GlobalTransform,
            depth: u8,
            index: VolumeIndex,
        ) -> Task<Self> {
            AsyncComputeTaskPool::get().spawn(async move {
                let mut voxels = [Voxel::default(); GRID_SIZE * GRID_SIZE * GRID_SIZE];

                for i in 0..(GRID_SIZE * GRID_SIZE * GRID_SIZE) {
                    // to get voxel centers, offset is half-width of chunk minus half-width of voxel
                    let offset = Vec3::splat(0.5 - 1. / 64.);
                    let [x, y, z] = [i % 32, i / 32 % 32, i / 32 / 32];
                    let position = Vec3::new(x as f32, y as f32, z as f32);
                    voxels[i as usize] = sampler(transform.transform_point(position - offset));
                }

                let empty = voxels
                    .iter()
                    .any(|voxel| voxel.opaque() != voxels[0].opaque());

                Self {
                    voxels,
                    sampler,
                    depth,
                    empty,
                    index,
                }
            })
        }

        /// `GridVolume`s should subdivide if they are close to viewers, have not reached maximum
        /// subdivision depth, and their region contains the surface (at voxel density zero)
        pub(crate) fn should_upsample<'a>(
            &self,
            transform: &GlobalTransform,
            mut viewers: impl Iterator<Item = (&'a Viewer, &'a GlobalTransform)>,
        ) -> bool {
            viewers.any(|(viewer, viewer_transform)| {
                viewer.should_upsample(viewer_transform, transform)
            }) && self.depth > 0
                && !self.empty
        }

        /// Create async functions which will generate children produced by an octree-style
        /// subdivision.
        pub(crate) fn child_tasks(
            &self,
            transform: &GlobalTransform,
        ) -> impl Iterator<Item = (Task<Self>, Transform)> {
            self.index.child_indices().map(|child_index| {
                (
                    Self::generate(
                        self.sampler.clone(),
                        transform.mul_transform(child_index.as_transform()),
                        self.depth - 1,
                        child_index,
                    ),
                    child_index.as_transform(),
                )
            })
        }

        /// Send all [`Cell`] data from this `GridVolume` that contains the surface, so that a
        /// [`Mesher`] can produce a [`Mesh`]. Use [`Subsection`] to specify a volume boundary
        /// along which to collect [`Cell`]s, rather than the whole volume.
        pub(crate) fn cells(&self, subsection: Subsection) -> Vec<([usize; 3], Cell)> {
            if self.empty {
                return Vec::new();
            }

            let [x_range, y_range, z_range] = subsection.to_ranges(GRID_SIZE, 1);

            iproduct!(x_range, y_range, z_range)
                .filter_map(|(x, y, z)| {
                    let mut voxels = [Voxel::default(); 8];
                    let mut contains_surface = false;
                    for (i, (x_offset, y_offset, z_offset)) in octants(0, 1).enumerate() {
                        let (x, y, z) = (x + x_offset, y + y_offset, z + z_offset);
                        voxels[i] = self.voxels[z * GRID_SIZE * GRID_SIZE + y * GRID_SIZE + x];
                        contains_surface |= voxels[0].opaque() != voxels[i].opaque();
                    }

                    if contains_surface {
                        Some(([x, y, z], Cell { voxels }))
                    } else {
                        None
                    }
                })
                .collect()
        }
    }
}
pub use volume::Volume;

/// # Volume Maps
mod volume_map {
    use super::{octants, GenerateTask, Volume, Voxel};
    use bevy::platform::collections::HashMap;
    use bevy::prelude::*;

    #[derive(Clone, Copy, Eq, Hash, PartialEq)]
    pub struct VolumeIndex {
        root: u32,
        x: u32,
        y: u32,
        z: u32,
        scale: u8,
    }

    impl VolumeIndex {
        pub(crate) fn root(root: u32) -> Self {
            Self {
                root,
                x: 0,
                y: 0,
                z: 0,
                scale: 0,
            }
        }

        pub(crate) fn new(root: u32, coordinates: [u32; 3], scale: u8) -> Self {
            Self {
                root,
                x: coordinates[0],
                y: coordinates[1],
                z: coordinates[2],
                scale,
            }
        }

        pub fn child_indices(&self) -> impl Iterator<Item = Self> {
            octants(0, 1).map(|(x, y, z)| Self {
                root: self.root,
                x: self.x * 2 + x,
                y: self.y * 2 + y,
                z: self.z * 2 + z,
                scale: self.scale + 1,
            })
        }

        pub fn as_transform(&self) -> Transform {
            let offset =
                UVec3::new(self.x % 2, self.y % 2, self.z % 2).as_vec3() * 0.5 - Vec3::splat(0.25);
            Transform::from_translation(offset).with_scale(Vec3::splat(0.5))
        }

        pub fn parent_index(&self) -> Option<Self> {
            if self.scale > 0 {
                Some(Self {
                    root: self.root,
                    x: self.x / 2,
                    y: self.y / 2,
                    z: self.z / 2,
                    scale: self.scale - 1,
                })
            } else {
                None
            }
        }

        pub fn offset(&self, x: u32, y: u32, z: u32) -> Self {
            Self {
                root: self.root,
                x: self.x + x,
                y: self.y + y,
                z: self.z + z,
                scale: self.scale,
            }
        }

        pub fn scale(&self) -> u8 {
            self.scale
        }
    }

    #[derive(Clone, Copy)]
    pub struct VolumeContext {
        pub entity: Entity,
        pub index: VolumeIndex,
        pub meshing: bool,
    }

    #[derive(Clone, Copy)]
    pub struct VolumeQuery {
        pub index: VolumeIndex,
        pub meshing: bool,
        pub coarse: bool,
    }

    impl VolumeQuery {
        fn matches(&self, flags: &VolumeContext) -> bool {
            self.meshing == flags.meshing
        }
    }

    #[derive(Default, Resource)]
    pub struct VolumeMap {
        volumes: HashMap<VolumeIndex, VolumeContext>,
        next_root: u32,
    }

    impl VolumeMap {
        pub fn new_root(
            &mut self,
            sampler: fn(Vec3) -> Voxel,
            transform: Transform,
            depth: u8,
        ) -> impl Bundle {
            let index = VolumeIndex::root(self.next_root);
            self.next_root += 1;
            (
                GenerateTask(Volume::generate(sampler, transform.into(), depth, index)),
                transform,
            )
        }

        pub(crate) fn insert(&mut self, volume_context: VolumeContext) {
            self.volumes.insert(volume_context.index, volume_context);
        }

        /// Returns the volume specified, or if not present but a matching volume one subdivision
        /// level higher exists, returns the higher subdivision volume.
        pub fn query(&self, query: VolumeQuery) -> Option<&VolumeContext> {
            if let Some(context) = self.volumes.get(&query.index).or(query
                .index
                .parent_index()
                .take_if(|_| query.coarse)
                .and_then(|index| self.volumes.get(&index)))
            {
                return query.matches(context).then_some(context);
            }
            None
        }
    }
}
pub use volume_map::{VolumeContext, VolumeIndex, VolumeMap, VolumeQuery};
