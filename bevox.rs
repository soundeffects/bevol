/*! # Voxel Plugin
Bevy plugin to manage a voxel world based on a sparse octree hierarchy for level-of-detail
rendering.

User-facing API:
- `Volume` components represent objects or regions of space that are filled in by voxel data
- `Volume` entites are positioned in the world with a `Transform` component
- `Volume`s are generic on a `Voxel` type, allowing users to store arbitrary spatial data
    - There are some provided Voxel types for convenience.
- `Volume`s recieve a fn(Vec3) -> Voxel, used for generation
- `Volume`s recieve a fn([Voxel; 8]) -> Vec3, used for placing vertices on the dual grid
    - There are a number of provided vertex placers, including "boxy", "smooth", and "contour"
- `Volume`s recieve a depth (u8), delimiting maximum hierarchy/subdivision depth
- Generation and meshing of volumes happen asynchronously.
- Camera entities with the `Viewer` component determine the focal points of level-of-detail.

Eventual goals for the API:
- Flat normals
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

use bevy::prelude::*;
use itertools::iproduct;

fn octants<T: Copy>(min: T, max: T) -> impl Iterator<Item = (T, T, T)> {
    iproduct!([min, max], [min, max], [min, max])
}

/// Camera entities with the `Viewer` component determine the focal points of level-of-detail.
#[derive(Component)]
pub struct Viewer {
    lod_distance: f32,
}

impl Viewer {
    pub fn new(lod_distance: f32) -> Self {
        Self { lod_distance }
    }

    pub fn should_upsample(
        &self,
        viewer_transform: &GlobalTransform,
        volume_transform: &GlobalTransform,
    ) -> bool {
        viewer_transform
            .translation()
            .distance(volume_transform.translation())
            < self.lod_distance
    }
}

/// # Placing
mod placing {
    use super::Voxel;
    use bevy::prelude::Vec3;

    pub fn boxy_placer<V: Voxel>(corner_voxels: [V; 8]) -> Vec3 {
        corner_voxels
            .iter()
            .position(|v| v.opaque())
            .map_or(Vec3::ZERO, |idx| {
                let x = (idx & 1) as f32 * 0.5;
                let y = ((idx >> 1) & 1) as f32 * 0.5;
                let z = ((idx >> 2) & 1) as f32 * 0.5;
                Vec3::new(x, y, z)
            })
    }

    pub fn smooth_placer<V: Voxel>(corner_voxels: [V; 8]) -> Vec3 {
        let sum: f32 = corner_voxels.iter().map(|v| v.density()).sum();
        let avg = sum / 8.0;
        Vec3::splat(avg * 0.5)
    }

    pub fn contour_placer<V: Voxel>(corner_voxels: [V; 8]) -> Vec3 {
        let has_opaque = corner_voxels.iter().any(|v| v.opaque());
        let has_transparent = corner_voxels.iter().any(|v| !v.opaque());

        if has_opaque && has_transparent {
            let opaque_count = corner_voxels.iter().filter(|v| v.opaque()).count() as f32;
            let t = opaque_count / 8.0;
            Vec3::splat(t * 0.5 - 0.25)
        } else if has_opaque {
            Vec3::splat(0.25)
        } else {
            Vec3::splat(-0.25)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{boxy_placer, contour_placer, smooth_placer, Voxel};
        use crate::StandardVoxel;

        #[test]
        fn test_placer_boxy() {
            let mut corners = [StandardVoxel::default(); 8];
            corners[0] = StandardVoxel::new(1, 1.0);
            let pos = boxy_placer(corners);
            assert!(pos.x >= 0.0 && pos.x <= 0.5);
        }

        #[test]
        fn test_placer_smooth() {
            let corners = [StandardVoxel::default(); 8];
            let pos = smooth_placer(corners);
            assert!(pos.x.is_finite());
            assert!(pos.y.is_finite());
            assert!(pos.z.is_finite());
        }

        #[test]
        fn test_placer_contoured() {
            let mut corners = [StandardVoxel::default(); 8];
            corners[0] = StandardVoxel::new(1, 1.0);
            corners[7] = StandardVoxel::new(1, -1.0);
            let pos = contour_placer(corners);
            assert!(pos.x >= -0.25 && pos.x <= 0.25);
        }
    }
}
pub use placing::{boxy_placer, contour_placer, smooth_placer};

/// # Voxels
mod voxel {
    /// Represents a single voxel in a [`Volume`](super::Volume). Implementors can store arbitrary
    /// data per-voxel while satisfying the constraints required for async generation and meshing.
    ///
    /// # Example
    /// ```
    /// # use bevox::{Voxel, StandardVoxel};
    /// // `bevox::StandardVoxel` is a provided implementer of `Voxel`
    /// let voxel = StandardVoxel::new(1, 1.0);
    /// assert!(voxel.opaque());
    /// ```
    pub trait Voxel: Clone + Copy + Default + Send + Sync {
        /// Returns the density value of this voxel, normalized to the range `[-1.0, 1.0]`, where
        /// `-1.0` represents empty space, `1.0` represents solid matter. The boundary between empty
        /// space and solid matter is determined by the implmementation of [`Voxel::opaque`].
        ///
        /// # Example
        /// ```
        /// # use bevox::{Voxel, StandardVoxel};
        /// assert!(StandardVoxel::new(0, 1.0).density() > 0.0);
        /// ```
        fn density(&self) -> f32;

        /// Returns `true` if this voxel should be considered solid (opaque) for mesh generation.
        ///
        /// The exact threshold depends on the implementation. For [`StandardVoxel`], voxels
        /// with density above 0.0 (midpoint) are opaque.
        ///
        /// # Example
        /// ```
        /// # use bevox::{Voxel, StandardVoxel};
        /// assert!(StandardVoxel::new(1, 1.0).opaque());
        /// ```
        fn opaque(&self) -> bool;
    }

    /// A simple voxel type storing a material identifier and a density value, provided for ease of
    /// use.
    ///
    /// # Example
    /// ```
    /// # use bevox::{Voxel, StandardVoxel};
    /// let example = StandardVoxel::new(0, 1.0);
    /// assert!(example.material_id() == 0);
    /// assert!(example.opaque());
    /// ```
    #[derive(Clone, Copy, Default)]
    pub struct StandardVoxel {
        /// An identifier for the material type of this voxel.
        pub material_id: u8,
        /// The internal density storage, normalized to `[0, 255]`.
        pub density: u8,
    }

    impl StandardVoxel {
        /// Creates a new `StandardVoxel` with the given material ID and density. The material ID
        /// should be within the range `[0, 255]` and the density will be clamped to the range
        /// `[-1.0, 1.0]`. Density values above `0.0` will be considered opaque.
        ///
        /// # Example
        /// ```
        /// # use bevox::{Voxel, StandardVoxel};
        /// let example = StandardVoxel::new(0, 1.0);
        /// assert!(example.material_id() == 0);
        /// assert!(example.opaque());
        /// ```
        pub fn new(material_id: u8, density: f32) -> Self {
            Self {
                material_id,
                density: ((density.clamp(-1.0, 1.0) + 1.0) * (u8::MAX as f32) / 2.0) as u8,
            }
        }

        /// Retrieves the material id of this voxel, which is stored as a `u8` type.
        ///
        /// # Example
        /// ```
        /// # use bevox::StandardVoxel;
        /// assert!(StandardVoxel::new(0, 0.0).material_id() == 0);
        /// ```
        pub fn material_id(&self) -> u8 {
            self.material_id
        }
    }

    impl Voxel for StandardVoxel {
        fn density(&self) -> f32 {
            let density = self.density as f32;
            let max = u8::MAX as f32;
            (density / max) * 2.0 - 1.0
        }

        fn opaque(&self) -> bool {
            self.density > u8::MAX / 2
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{StandardVoxel, Voxel};

        #[test]
        fn test_standard_voxel_new() {
            let voxel = StandardVoxel::new(1, 0.0);
            assert_eq!(voxel.material_id, 1);
            assert_eq!(voxel.density, u8::MAX / 2);
        }

        #[test]
        fn test_standard_voxel_density() {
            let voxel_max = StandardVoxel::new(0, 1.0);
            assert!((voxel_max.density() - 1.0).abs() < 0.01);

            let voxel_min = StandardVoxel::new(0, -1.0);
            assert!((voxel_min.density() - (-1.0)).abs() < 0.01);
        }

        #[test]
        fn test_standard_voxel_opaque() {
            let opaque_voxel = StandardVoxel::new(0, 1.0);
            assert!(opaque_voxel.opaque());

            let transparent_voxel = StandardVoxel::new(0, -1.0);
            assert!(!transparent_voxel.opaque());
        }
    }
}
pub use voxel::{StandardVoxel, Voxel};

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

    #[cfg(test)]
    mod tests {
        use super::Subsection;

        #[test]
        fn test_subsection_to_ranges_none() {
            let sub = Subsection::None;
            let ranges = sub.to_ranges(32, 1);
            assert_eq!(ranges[0].start, 0);
            assert!(ranges[0].end > 0);
        }

        #[test]
        fn test_subsection_to_ranges_xyz_corner() {
            let sub = Subsection::XYZCorner;
            let ranges = sub.to_ranges(32, 1);
            assert_eq!(ranges[0].start, 0);
            assert_eq!(ranges[0].end, 1);
            assert_eq!(ranges[1].start, 0);
            assert_eq!(ranges[1].end, 1);
            assert_eq!(ranges[2].start, 0);
            assert_eq!(ranges[2].end, 1);
        }
    }
}
use subsection::{Half, Quadrant, Subsection};

/// # Volumes
mod volume {
    use super::{octants, Viewer, Voxel};
    use bevy::{
        asset::RenderAssetUsages,
        mesh::{Indices, PrimitiveTopology},
        prelude::*,
        tasks::{AsyncComputeTaskPool, Task},
    };

    /// A task handle for an asynchronously generated [`Volume`]. When polled, the task will
    /// complete the voxel generation and return the fully populated [`Volume`].
    ///
    /// # Example
    /// ```
    /// # use bevy::prelude::*;
    /// # use bevox::{StandardVolume, StandardVoxel, boxy_placer};
    /// # fn sampler(pos: Vec3) -> StandardVoxel { unimplemented!() }
    /// fn my_bevy_system(mut commands: Commands) {
    ///     // The following spawns a `GenerateTask`
    ///     commands.spawn(StandardVolume::<StandardVoxel>::generate(
    ///         // See `Volume::generate` for args...
    ///         # sampler,
    ///         # boxy_placer,
    ///         # GlobalTransform::IDENTITY,
    ///         # 0
    ///     ));
    /// }
    /// ```
    #[derive(Component)]
    pub struct GenerateTask<V: Voxel + 'static, const N: usize>(Task<Volume<V, N>>);

    /// A task handle for an asynchronously generated [`Mesh`]. When polled, the task will
    /// complete the mesh generation and return a Bevy [`Mesh`]. The mesh is produced using a
    /// dual-contouring algorithm, with vertex placement determined by a function passed in by the
    /// user when creating a [`Volume`] using [`Volume::generate`].
    #[derive(Component)]
    pub(crate) struct MeshTask(Task<Mesh>);

    /// A cubical grid of voxels, also known as a "chunk". Will automatically be subdivided
    /// (octree-style) by [`VoxelsPlugin`] systems.
    ///
    /// - Stores a sampling function to determine voxel values at generation, which it will pass on
    ///   to any subdivided children.
    /// - Stores a placing function which will place vertices on the dual grid when meshing the
    ///   volume.
    /// - Stores a depth to limit the number of subdivisions this volume will have.
    /// - Is generic on the `Voxel` type stored
    /// - Is generic on the side length `N` of the cubical region the `Volume` governs
    ///
    /// # Example
    /// ```
    /// # use bevy::prelude::*;
    /// # use bevox::{Volume, StandardVoxel, boxy_placer};
    /// # fn sampler(pos: Vec3) -> StandardVoxel { unimplemented!() }
    /// fn my_bevy_system(mut commands: Commands) {
    ///     commands.spawn(Volume::<StandardVoxel, 16>::generate(
    ///         /* See `Volume::generate` for args...*/
    ///         # sampler,
    ///         # boxy_placer,
    ///         # GlobalTransform::IDENTITY,
    ///         # 0
    ///     ));
    /// }
    /// ```
    #[derive(Clone, Component)]
    #[require(GlobalTransform)]
    pub struct Volume<V: Voxel + 'static, const N: usize> {
        voxels: [[[V; N]; N]; N],
        sampler: fn(Vec3) -> V,
        placer: fn([V; 8]) -> Vec3,
        depth: u8,
    }

    /// This is a [`Volume`] with a set cubical length of 16 voxels. It can be used in all the same
    /// ways as a [`Volume`]; see [`Volume`] for more info.
    pub type StandardVolume<V> = Volume<V, 16>;

    impl<V: Voxel + 'static, const N: usize> Volume<V, N> {
        /// Generate a new `Volume` using the following:
        /// - A 3D sampling function to determine voxel values
        /// - A vertex-placing function used to place vertices on the dual grid when meshing
        /// - A [`GlobalTransform`] to signal where this `Volume` is placed
        /// - A depth used to limit the number of subdivisions
        /// - Specify the generic [`Voxel`] and `N` (volume size)
        ///
        /// Returns a [`GenerateTask`] component which will asynchronously produce the `Volume`.
        ///
        /// # Example
        /// ```
        /// # use bevox::{Volume, StandardVoxel, boxy_placer};
        /// # use bevy::prelude::*;
        /// #
        /// // Simple terrain sampler: solid below y=0
        /// fn terrain_sampler(pos: Vec3) -> StandardVoxel {
        ///     if pos.y < 0.0 {
        ///         StandardVoxel::new(1, 1.0)
        ///     } else {
        ///         StandardVoxel::new(2, -1.0)
        ///     }
        /// }
        ///
        /// fn my_bevy_system(mut commands: Commands) {
        ///     commands.spawn(Volume::<StandardVoxel, 16>::generate(
        ///         terrain_sampler,
        ///         bevox::boxy_placer,
        ///         GlobalTransform::from_translation(Vec3::ZERO),
        ///         4 // depth of subdivisions
        ///     ));
        /// }
        /// ```
        pub fn generate(
            sampler: fn(Vec3) -> V,
            placer: fn([V; 8]) -> Vec3,
            transform: GlobalTransform,
            depth: u8,
        ) -> GenerateTask<V, N> {
            let task = AsyncComputeTaskPool::get().spawn(async move {
                let mut voxels = [[[V::default(); N]; N]; N];

                #[allow(clippy::needless_range_loop)]
                for z in 0..N {
                    for y in 0..N {
                        for x in 0..N {
                            let offset = Vec3::splat(0.5 - 1. / (2.0 * N as f32));
                            let position = Vec3::new(x as f32, y as f32, z as f32);
                            voxels[z][y][x] = sampler(transform.transform_point(position - offset));
                        }
                    }
                }

                Self {
                    voxels,
                    sampler,
                    placer,
                    depth,
                }
            });
            GenerateTask(task)
        }

        /// Volume should subdivide if they are close to viewers, have not reached maximum
        /// subdivision depth.
        pub(crate) fn should_upsample<'a>(
            &self,
            transform: &GlobalTransform,
            mut viewers: impl Iterator<Item = (&'a Viewer, &'a GlobalTransform)>,
        ) -> bool {
            viewers.any(|(viewer, viewer_transform)| {
                viewer.should_upsample(viewer_transform, transform)
            }) && self.depth > 0
        }

        /// Create a set async tasks to generate eight subdivided children of this volume, each
        /// governing an octant of this volume's cubical region. Provide the transformation
        /// representing the offset with each task.
        pub(crate) fn child_tasks(
            &self,
            transform: &GlobalTransform,
        ) -> impl Iterator<Item = (GenerateTask<V, N>, Transform)> {
            (0..8).map(move |i| {
                let child_transform = transform.mul_transform(
                    Transform::from_translation(Vec3::new(
                        (i & 1) as f32 * 0.5 - 0.25,
                        ((i >> 1) & 1) as f32 * 0.5 - 0.25,
                        ((i >> 2) & 1) as f32 * 0.5 - 0.25,
                    ))
                    .with_scale(Vec3::splat(0.5)),
                );
                (
                    Self::generate(self.sampler, self.placer, child_transform, self.depth - 1),
                    Transform::from_translation(Vec3::new(
                        (i & 1) as f32 * 0.5 - 0.25,
                        ((i >> 1) & 1) as f32 * 0.5 - 0.25,
                        ((i >> 2) & 1) as f32 * 0.5 - 0.25,
                    ))
                    .with_scale(Vec3::splat(0.5)),
                )
            })
        }

        /// Produces a [`Mesh`] for this `Volume` asynchronously, using the following steps:
        /// 1. Copies data voxel data from this `Volume`, plus the bordering edges of neighboring
        ///    `Volume`s in the positive X, Y, and Z directions.
        ///     - Note that this `Volume` must determine its neighbors by checking its own
        ///       transform to find what offset it was given, and collecting the siblings from its
        ///       own parent and the children of other higher-level `Volume`s at the appropriate
        ///       indices. Bevy keeps all children in the `Children` component in the order they
        ///       were added.
        /// 2. Creates a Bevy Task with the copied data to perform all following steps
        /// 3. Iterates through all `Cell`s of `Voxel`s, where every eight `Voxel`s arranged
        ///    cubically in the `Volume` and bordering edges are considered a `Cell` (Note that
        ///    `Cells` represent the "dual" of the voxel grid).
        /// 4. Produces a vertex position for every `Cell`, by providing the data to the `Volume`s
        ///    vertex placing function.
        /// 5. Writes indices for all vertices to create triangles between adjacent vertices.
        /// 6. Produces a Bevy [`Mesh`] and uses the provided method to automatically compute
        ///    normals
        pub(crate) fn mesh(&self, _volumes: Query<&Volume<V, N>>) -> MeshTask {
            let placer = self.placer;
            let center = self.voxels;
            let x_face = todo!();
            let y_face = todo!();
            let z_face = todo!();
            let xy_edge = todo!();
            let xz_edge = todo!();
            let yz_edge = todo!();
            let corner = todo!();

            MeshTask(AsyncComputeTaskPool::get().spawn(async move {
                let mut mesh_voxels = vec![V::default(); (N + 1) * (N + 1) * (N + 1)];

                for z in 0..N {
                    for y in 0..N {
                        for x in 0..N {
                            let dst_idx = z * (N + 1) * (N + 1) + y * (N + 1) + x;
                            mesh_voxels[dst_idx] = center[z][y][x];
                        }
                    }
                }

                let mut positions = Vec::new();
                let mut indices = Vec::new();

                for z in 0..N {
                    for y in 0..N {
                        for x in 0..N {
                            let base_idx = z * (N + 1) * (N + 1) + y * (N + 1) + x;
                            let corners = [
                                mesh_voxels[base_idx],
                                mesh_voxels[base_idx + 1],
                                mesh_voxels[base_idx + (N + 1)],
                                mesh_voxels[base_idx + (N + 1) + 1],
                                mesh_voxels[base_idx + (N + 1) * (N + 1)],
                                mesh_voxels[base_idx + (N + 1) * (N + 1) + 1],
                                mesh_voxels[base_idx + (N + 1) * (N + 1) + (N + 1)],
                                mesh_voxels[base_idx + (N + 1) * (N + 1) + (N + 1) + 1],
                            ];

                            let vertex_pos = placer(corners);
                            positions.push(vertex_pos);
                        }
                    }
                }

                for z in 0..N {
                    for y in 0..N {
                        for x in 0..N {
                            let i = (z * N + y) * N + x;

                            if x < N - 1 {
                                let i_next = i + 1;
                                indices.push(i as u32);
                                indices.push(i_next as u32);
                                indices.push((i + N) as u32);
                                indices.push(i_next as u32);
                                indices.push((i + N + 1) as u32);
                                indices.push((i + N) as u32);
                            }

                            if y < N - 1 {
                                let i_up = i + N;
                                indices.push(i as u32);
                                indices.push((i + 1) as u32);
                                indices.push(i_up as u32);
                                indices.push((i + 1) as u32);
                                indices.push((i + N + 1) as u32);
                                indices.push(i_up as u32);
                            }

                            if z < N - 1 {
                                let i_front = i + N * N;
                                indices.push(i as u32);
                                indices.push(i_front as u32);
                                indices.push((i + 1) as u32);
                                indices.push((i + 1) as u32);
                                indices.push(i_front as u32);
                                indices.push((i + N + 1) as u32);
                            }
                        }
                    }
                }

                Mesh::new(
                    PrimitiveTopology::TriangleList,
                    RenderAssetUsages::default(),
                )
                .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
                .with_inserted_indices(Indices::U32(indices))
                .with_computed_normals()
            }))
        }

        fn copy_local_region(
            self_entity: Entity,
            hierarchy: Query<(Entity, Option<&ChildOf>)>,
            volumes: Query<(&Volume<V, N>, &GlobalTransform)>,
        ) -> [Vec<V>; 8] {
            todo!()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{GenerateTask, StandardVolume, Viewer, Volume, Voxel};
        use crate::StandardVoxel;
        use bevy::prelude::*;

        fn test_sampler(_pos: Vec3) -> StandardVoxel {
            StandardVoxel::new(1, 1.0)
        }

        fn test_placer<V: Voxel>(_corners: [V; 8]) -> Vec3 {
            Vec3::splat(0.5)
        }

        #[test]
        fn test_volume_derives() {
            fn assert_clone<T: Clone>() {}
            fn assert_component<T: Component>() {}
            assert_clone::<Volume<StandardVoxel, 16>>();
            assert_component::<Volume<StandardVoxel, 16>>();
        }

        #[test]
        fn test_standard_volume_type_alias() {
            fn _assert<V: Voxel + 'static>() -> StandardVolume<V> {
                unimplemented!()
            }
        }

        #[test]
        #[ignore = "Async testing is not setup yet"]
        fn test_generate_creates_volume_with_correct_depth() {
            let transform = GlobalTransform::from_translation(Vec3::ONE);
            let depth = 4u8;
            let _task: GenerateTask<StandardVoxel, 16> =
                Volume::generate(test_sampler, test_placer, transform, depth);
        }

        #[test]
        #[ignore = "Async testing is not setup yet"]
        fn test_volume_default_voxel_size() {
            let transform = GlobalTransform::IDENTITY;
            let _task: GenerateTask<StandardVoxel, 8> =
                Volume::generate(test_sampler, test_placer, transform, 2);
        }

        #[test]
        fn test_should_upsample_far_viewer() {
            let volume_transform = GlobalTransform::from_translation(Vec3::ZERO);
            let volume = Volume {
                voxels: [[[StandardVoxel::default(); 8]; 8]; 8],
                sampler: test_sampler,
                placer: test_placer,
                depth: 1,
            };

            let viewer_transform_far = GlobalTransform::from_translation(Vec3::splat(1000.0));
            let viewer_far = Viewer::new(10.0);
            let viewers_far = std::iter::once((&viewer_far, &viewer_transform_far));
            assert!(!volume.should_upsample(&volume_transform, viewers_far));
        }

        #[test]
        fn test_should_upsample_close_viewer() {
            let volume_transform = GlobalTransform::from_translation(Vec3::new(10.0, 0.0, 0.0));
            let volume = Volume {
                voxels: [[[StandardVoxel::default(); 8]; 8]; 8],
                sampler: test_sampler,
                placer: test_placer,
                depth: 1,
            };

            let viewer_transform_near =
                GlobalTransform::from_translation(Vec3::new(10.5, 0.0, 0.0));
            let viewer_near = Viewer::new(10.0);
            let viewers_near = std::iter::once((&viewer_near, &viewer_transform_near));
            assert!(volume.should_upsample(&volume_transform, viewers_near));
        }

        #[test]
        fn test_should_upsample_limited_by_depth() {
            let volume_transform = GlobalTransform::from_translation(Vec3::ZERO);
            let volume = Volume {
                voxels: [[[StandardVoxel::default(); 8]; 8]; 8],
                sampler: test_sampler,
                placer: test_placer,
                depth: 0,
            };

            let viewer_transform_near = GlobalTransform::from_translation(Vec3::splat(0.5));
            let viewer_near = Viewer::new(100.0);
            let viewers_near = std::iter::once((&viewer_near, &viewer_transform_near));
            assert!(!volume.should_upsample(&volume_transform, viewers_near));
        }

        #[test]
        fn test_should_upsample_multiple_viewers() {
            let volume_transform = GlobalTransform::from_translation(Vec3::ONE);
            let volume = Volume {
                voxels: [[[StandardVoxel::default(); 8]; 8]; 8],
                sampler: test_sampler,
                placer: test_placer,
                depth: 1,
            };

            let viewer_far = Viewer::new(10.0);
            let viewer_near = Viewer::new(10.0);
            let viewers = [
                (
                    &viewer_far,
                    &GlobalTransform::from_translation(Vec3::splat(1000.0)),
                ),
                (
                    &viewer_near,
                    &GlobalTransform::from_translation(Vec3::splat(1.5)),
                ),
            ];
            let viewers_iter = viewers.into_iter();
            assert!(volume.should_upsample(&volume_transform, viewers_iter));
        }

        #[test]
        #[ignore = "Async testing is not setup yet"]
        fn test_child_tasks_produces_eight_children() {
            let transform = GlobalTransform::IDENTITY;
            let volume = Volume {
                voxels: [[[StandardVoxel::default(); 8]; 8]; 8],
                sampler: test_sampler,
                placer: test_placer,
                depth: 3,
            };

            let children: Vec<_> = volume.child_tasks(&transform).collect();
            assert_eq!(children.len(), 8);
        }

        #[test]
        #[ignore = "Async testing is not setup yet"]
        fn test_child_tasks_have_half_scale() {
            let transform = GlobalTransform::IDENTITY;
            let volume = Volume {
                voxels: [[[StandardVoxel::default(); 8]; 8]; 8],
                sampler: test_sampler,
                placer: test_placer,
                depth: 2,
            };

            let children: Vec<_> = volume.child_tasks(&transform).collect();
            for (_, child_transform) in children {
                assert_eq!(child_transform.scale, Vec3::splat(0.5));
            }
        }

        #[test]
        #[ignore = "Async testing is not setup yet"]
        fn test_child_tasks_have_correct_offsets() {
            let volume = Volume {
                voxels: [[[StandardVoxel::default(); 8]; 8]; 8],
                sampler: test_sampler,
                placer: test_placer,
                depth: 2,
            };

            let children: Vec<_> = volume.child_tasks(&GlobalTransform::IDENTITY).collect();
            let expected_offsets = [
                Vec3::new(-0.25, -0.25, -0.25),
                Vec3::new(0.25, -0.25, -0.25),
                Vec3::new(-0.25, 0.25, -0.25),
                Vec3::new(0.25, 0.25, -0.25),
                Vec3::new(-0.25, -0.25, 0.25),
                Vec3::new(0.25, -0.25, 0.25),
                Vec3::new(-0.25, 0.25, 0.25),
                Vec3::new(0.25, 0.25, 0.25),
            ];

            for (translation, expected) in children
                .iter()
                .map(|(_, transform)| transform.translation)
                .zip(expected_offsets.into_iter())
            {
                assert_eq!(translation, expected)
            }
        }

        #[test]
        #[ignore = "Async testing is not setup yet"]
        fn test_child_tasks_decrease_depth() {
            let transform = GlobalTransform::IDENTITY;
            let initial_depth = 5u8;
            let volume = Volume {
                voxels: [[[StandardVoxel::default(); 8]; 8]; 8],
                sampler: test_sampler,
                placer: test_placer,
                depth: initial_depth,
            };

            let children: Vec<_> = volume.child_tasks(&transform).collect();
            assert_eq!(children.len(), 8);
        }
    }
}
use volume::MeshTask;
pub use volume::{GenerateTask, StandardVolume, Volume};
