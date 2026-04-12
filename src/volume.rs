/*! # Volume Module
*/

use super::{mesh_cells, Cell, MeshTask, MeshTaskError, Viewer, Voxel};
use bevy::{
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task},
};
use itertools::iproduct;
use std::sync::{Arc, RwLock};

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
///         # 0,
///         false
///     ));
/// }
/// ```
#[derive(Component)]
pub struct GenerateTask<V: Voxel + 'static, const N: usize>(Task<Volume<V, N>>);

impl<V: Voxel, const N: usize> GenerateTask<V, N> {
    pub(crate) fn task_mut(&mut self) -> &mut Task<Volume<V, N>> {
        &mut self.0
    }
}

/// A cubical grid of voxels, also known as a "chunk". Will automatically be subdivided
/// (octree-style) by [`VoxelsPlugin`] systems.
///
/// - Stores a sampling function to determine voxel values at generation, which it will pass on
///   to any subdivided children.
/// - Stores a placing function which will place vertices on the dual grid when meshing the
///   volume.
/// - Stores a depth to limit the number of subdivisions this volume will have.
/// - Stores whether this volume is trivial, meaning it does not intersect with the isosurface.
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
///         # 0,
///         false
///     ));
/// }
/// ```
#[derive(Clone, Component)]
#[require(GlobalTransform)]
pub struct Volume<V: Voxel + 'static, const N: usize> {
    voxel_store: Arc<RwLock<[[[V; N]; N]; N]>>,
    sampler: fn(Vec3) -> V,
    placer: fn(Cell<V>) -> Vec3,
    depth: u8,
    trivial_opacity: Option<bool>,
    flat_normals: bool,
}

/// This is a [`Volume`] with a set cubical length of 16 voxels. It can be used in all the same
/// ways as a [`Volume`]; see [`Volume`] for more info.
pub type StandardVolume<V> = Volume<V, 16>;

impl<V: Voxel + 'static, const N: usize> Volume<V, N> {
    pub(crate) fn from_voxel_store(
        voxel_store: Arc<RwLock<[[[V; N]; N]; N]>>,
        sampler: fn(Vec3) -> V,
        placer: fn(Cell<V>) -> Vec3,
        depth: u8,
        trivial_opacity: Option<bool>,
        flat_normals: bool,
    ) -> Self {
        Self {
            voxel_store,
            sampler,
            placer,
            depth,
            trivial_opacity,
            flat_normals,
        }
    }
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
    ///         4, // depth of subdivisions
    ///         false
    ///     ));
    /// }
    /// ```
    pub fn generate(
        sampler: fn(Vec3) -> V,
        placer: fn(Cell<V>) -> Vec3,
        transform: GlobalTransform,
        depth: u8,
        flat_normals: bool,
    ) -> GenerateTask<V, N> {
        let task = AsyncComputeTaskPool::get().spawn(async move {
            let mut voxels = [[[V::default(); N]; N]; N];

            for (x, y, z) in iproduct!(0..N, 0..N, 0..N) {
                let offset = Vec3::splat(0.5 - 1. / (2.0 * N as f32));
                let position = Vec3::new(x as f32, y as f32, z as f32);
                voxels[x][y][z] = sampler(transform.transform_point(position - offset));
            }

            let trivial_opacity = iproduct!(0..N, 0..N, 0..N)
                .map(|(x, y, z)| voxels[x][y][z])
                .all(|voxel| voxel.opaque() == voxels[0][0][0].opaque())
                .then_some(voxels[0][0][0].opaque());

            Self {
                voxel_store: Arc::new(RwLock::new(voxels)),
                sampler,
                placer,
                depth,
                trivial_opacity,
                flat_normals,
            }
        });
        GenerateTask(task)
    }

    /// Volume should subdivide if they are close to viewers, have not reached maximum
    /// subdivision depth, and are not a trivial volume with no surface cells.
    pub(crate) fn should_upsample<'a>(
        &self,
        transform: &GlobalTransform,
        mut viewers: impl Iterator<Item = (&'a Viewer, &'a GlobalTransform)>,
    ) -> bool {
        viewers
            .any(|(viewer, viewer_transform)| viewer.should_upsample(viewer_transform, transform))
            && self.depth > 0
            && self.trivial_opacity.is_none()
    }

    /// Create a set async tasks to generate eight subdivided children of this volume, each
    /// governing an octant of this volume's cubical region. Provide the transformation
    /// representing the offset with each task.
    pub(crate) fn child_tasks(
        &self,
        transform: &GlobalTransform,
    ) -> impl Iterator<Item = (GenerateTask<V, N>, Transform)> {
        (0..8).map(move |i| {
            let relative_transform = Transform::from_translation(Vec3::new(
                (i & 1) as f32 * 0.5 - 0.25,
                ((i >> 1) & 1) as f32 * 0.5 - 0.25,
                ((i >> 2) & 1) as f32 * 0.5 - 0.25,
            ))
            .with_scale(Vec3::splat(0.5));
            let child_transform = transform.mul_transform(relative_transform);
            (
                Self::generate(
                    self.sampler,
                    self.placer,
                    child_transform,
                    self.depth - 1,
                    self.flat_normals,
                ),
                relative_transform,
            )
        })
    }

    /// Produces a [`Mesh`] for this `Volume` asynchronously. This mesh will only contain
    /// interior surface data, and not any border regions where the `Volume` interfaces with
    /// neighboring `Volume`s.
    pub(crate) fn mesh(&self) -> MeshTask {
        let placer = self.placer;
        let flat_normals = self.flat_normals;
        let voxel_lock = self.voxel_store.clone();

        MeshTask::new(async move {
            match voxel_lock.read() {
                Ok(voxels) => Ok(mesh_cells(
                    iproduct!(0..(N - 1), 0..(N - 1), 0..(N - 1)).map(|(x, y, z)| {
                        (
                            UVec3::new(x as u32, y as u32, z as u32),
                            Cell {
                                nx_ny_nz: voxels[x][y][z],
                                nx_ny_pz: voxels[x][y][z + 1],
                                nx_py_nz: voxels[x][y + 1][z],
                                nx_py_pz: voxels[x][y + 1][z + 1],
                                px_ny_nz: voxels[x + 1][y][z],
                                px_ny_pz: voxels[x + 1][y][z + 1],
                                px_py_nz: voxels[x + 1][y + 1][z],
                                px_py_pz: voxels[x + 1][y + 1][z + 1],
                            },
                        )
                    }),
                    placer,
                    flat_normals,
                )),
                Err(_) => Err(MeshTaskError::PoisonedRwLock),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{Cell, GenerateTask, StandardVolume, Viewer, Volume, Voxel};
    use crate::StandardVoxel;
    use bevy::prelude::*;
    use std::sync::{Arc, RwLock};

    fn test_sampler(_pos: Vec3) -> StandardVoxel {
        StandardVoxel::new(1, 1.0)
    }

    fn test_placer<V: Voxel>(cell: Cell<V>) -> Vec3 {
        let _ = cell;
        Vec3::splat(0.5)
    }

    fn make_test_volume(depth: u8) -> Volume<StandardVoxel, 8> {
        let voxels = [[[StandardVoxel::default(); 8]; 8]; 8];
        let voxel_store = Arc::new(RwLock::new(voxels));
        Volume::from_voxel_store(voxel_store, test_sampler, test_placer, depth, None, false)
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
            Volume::generate(test_sampler, test_placer, transform, depth, false);
    }

    #[test]
    #[ignore = "Async testing is not setup yet"]
    fn test_volume_default_voxel_size() {
        let transform = GlobalTransform::IDENTITY;
        let _task: GenerateTask<StandardVoxel, 8> =
            Volume::generate(test_sampler, test_placer, transform, 2, false);
    }

    #[test]
    fn test_should_upsample_far_viewer() {
        let volume_transform = GlobalTransform::from_translation(Vec3::ZERO);
        let volume = make_test_volume(1);

        let viewer_transform_far = GlobalTransform::from_translation(Vec3::splat(1000.0));
        let viewer_far = Viewer::new(10.0);
        let viewers_far = std::iter::once((&viewer_far, &viewer_transform_far));
        assert!(!volume.should_upsample(&volume_transform, viewers_far));
    }

    #[test]
    fn test_should_upsample_close_viewer() {
        let volume_transform = GlobalTransform::from_translation(Vec3::new(10.0, 0.0, 0.0));
        let volume = make_test_volume(1);

        let viewer_transform_near = GlobalTransform::from_translation(Vec3::new(10.5, 0.0, 0.0));
        let viewer_near = Viewer::new(10.0);
        let viewers_near = std::iter::once((&viewer_near, &viewer_transform_near));
        assert!(volume.should_upsample(&volume_transform, viewers_near));
    }

    #[test]
    fn test_should_upsample_limited_by_depth() {
        let volume_transform = GlobalTransform::from_translation(Vec3::ZERO);
        let volume = make_test_volume(0);

        let viewer_transform_near = GlobalTransform::from_translation(Vec3::splat(0.5));
        let viewer_near = Viewer::new(100.0);
        let viewers_near = std::iter::once((&viewer_near, &viewer_transform_near));
        assert!(!volume.should_upsample(&volume_transform, viewers_near));
    }

    #[test]
    fn test_should_upsample_multiple_viewers() {
        let volume_transform = GlobalTransform::from_translation(Vec3::ONE);
        let volume = make_test_volume(1);

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
        let volume = make_test_volume(3);

        let children: Vec<_> = volume.child_tasks(&transform).collect();
        assert_eq!(children.len(), 8);
    }

    #[test]
    #[ignore = "Async testing is not setup yet"]
    fn test_child_tasks_have_half_scale() {
        let transform = GlobalTransform::IDENTITY;
        let volume = make_test_volume(2);

        let children: Vec<_> = volume.child_tasks(&transform).collect();
        for (_, child_transform) in children {
            assert_eq!(child_transform.scale, Vec3::splat(0.5));
        }
    }

    #[test]
    #[ignore = "Async testing is not setup yet"]
    fn test_child_tasks_have_correct_offsets() {
        let volume = make_test_volume(2);

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
        let volume = make_test_volume(initial_depth);

        let children: Vec<_> = volume.child_tasks(&transform).collect();
        assert_eq!(children.len(), 8);
    }
}
