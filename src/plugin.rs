/*! # Plugin Module
Provides Bevy [`Plugin`]s and systems which handle:
- LOD (Level of Detail) management through the [`Viewer`] component
- Polling async generation and meshing tasks
- Mesh material setup for rendering voxel meshes
*/

use super::{GenerateTask, MeshTask, Volume, Voxel};
use bevy::{
    ecs::entity_disabling::Disabled, platform::collections::HashSet, prelude::*,
    tasks::futures::check_ready,
};

/// Camera entities with the `Viewer` component determine the focal points of level-of-detail. When
/// a camera has `Viewer` component, the plugin will automatically subdivide all [`Volume`]s within
/// a distance of the `Viewer`s limit (provided on instantiation), divided by the [`Volume`]s
/// scale. The plugin will also merge any subdivided volumes that fall outside of this limit.
///
/// # Example
/// ```
/// # use bevox::Viewer;
/// # use bevy::prelude::*;
/// fn setup_camera(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
///     commands.spawn((
///         Camera3d::default(),
///         Transform::from_translation(Vec3::new(0.0, 10.0, 20.0)),
///         Viewer::new(15.0), // LOD distance of 15 units
///     ));
/// }
/// ```
#[derive(Component)]
pub struct Viewer {
    lod_distance: f32,
}

impl Viewer {
    /// Creates a new Viewer with the specified LOD distance. See [`Viewer`] for more info.
    ///
    /// # Example
    /// ```
    /// # use bevox::Viewer;
    /// # use bevy::prelude::*;
    /// fn setup_camera(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    ///     commands.spawn((
    ///         Camera3d::default(),
    ///         Transform::from_translation(Vec3::new(0.0, 10.0, 20.0)),
    ///         Viewer::new(15.0), // LOD distance of 15 units
    ///     ));
    /// }
    /// ```
    pub fn new(lod_distance: f32) -> Self {
        Self { lod_distance }
    }

    /// Determines whether a volume should be subdivided (upsampled) based on the viewer's distance
    /// to it. This is used internally to decide whether to spawn child volumes or keep the current
    /// level of detail.
    pub(crate) fn should_upsample(
        &self,
        viewer_transform: &GlobalTransform,
        volume_transform: &GlobalTransform,
    ) -> bool {
        viewer_transform
            .translation()
            .distance(volume_transform.translation())
            * viewer_transform.scale().max_element()
            <= self.lod_distance
    }
}

/// The main Bevox plugin for Bevy, managing voxel world generation and rendering. Adding this
/// plugin to the Bevy app, and specifying a `V` voxel type and `N` volume size with
/// [`BevoxPlugin::for_parameters`] or [`BevoxPlugin::and_parameters`], will manage all [`Volume`]s
/// with the same `V` and `N` parameters.
///
/// # Example
/// ```
/// # use bevy::prelude::*;
/// # use bevox::{BevoxPlugin, StandardVoxel};
/// App::new().add_plugins(
///     BevoxPlugin::for_parameters::<StandardVoxel, 16>().and_parameters::<StandardVoxel, 32>()
/// );
/// ```
///
/// # Plugin Dependencies
/// This plugin requires the [`MinimalPlugins`] set, as well as the following plugins:
/// - [`AssetPlugin`]
/// - [`ImagePlugin`]
/// - [`bevy::render::RenderPlugin`]
/// - [`bevy::core_pipeline::CorePipelinePlugin`]
/// - [`bevy::mesh::MeshPlugin`]
/// - [`bevy::pbr::PbrPlugin`]
pub struct BevoxPlugin {
    generic_handlers: Vec<fn(&mut App)>,
}

impl BevoxPlugin {
    /// Allows adding systems that require generic paramters in a delayed/type-erased manner.
    fn handler<V: Voxel + 'static, const N: usize>(app: &mut App) {
        app.add_systems(
            Update,
            (poll_generate_task::<V, N>, resample_volumes::<V, N>),
        );
    }

    /// Creates a new [`BevoxPlugin`], and tells it to manage [`Volume`]s of the `V` voxel type and
    /// size `N`. See [`BevoxPlugin`] for more info.
    ///
    /// # Example
    /// ```
    /// # use bevy::prelude::*;
    /// # use bevox::{BevoxPlugin, StandardVoxel};
    /// App::new().add_plugins(
    ///     BevoxPlugin::for_parameters::<StandardVoxel, 16>()
    /// );
    /// ```
    pub fn for_parameters<V: Voxel + 'static, const N: usize>() -> Self {
        Self {
            generic_handlers: vec![Self::handler::<V, N>],
        }
    }

    /// If a [`BevoxPlugin`] has already been created using [`BevoxPlugin::for_parameters`], you
    /// can specify additional parameter sets for the plugin to handle by calling this method after
    /// creation. See [`BevoxPlugin`] for more info.
    ///
    /// # Example
    /// ```
    /// # use bevy::prelude::*;
    /// # use bevox::{BevoxPlugin, StandardVoxel};
    /// App::new().add_plugins(
    ///     BevoxPlugin::for_parameters::<StandardVoxel, 16>()
    ///         .and_parameters::<StandardVoxel, 32>()
    /// );
    /// ```
    pub fn and_parameters<V: Voxel + 'static, const N: usize>(mut self) -> Self {
        self.generic_handlers.push(Self::handler::<V, N>);
        self
    }
}

impl Plugin for BevoxPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_voxel_material)
            .add_systems(Update, poll_mesh_task);
        for handler in self.generic_handlers.iter() {
            handler(app);
        }
    }
}

/// The default material used for rendering voxel meshes.
#[derive(Resource)]
pub(crate) struct VoxelMaterial(Handle<StandardMaterial>);

/// Adds the default [`VoxelMaterial`] during startup.
fn setup_voxel_material(mut commands: Commands, mut materials: ResMut<Assets<StandardMaterial>>) {
    commands.insert_resource(VoxelMaterial(materials.add(StandardMaterial::default())));
}

/// If any async [`GenerateTask`]s have completed, this system will collect the generated
/// [`Volume`] component and add it to the associated entity.
fn poll_generate_task<V: Voxel, const N: usize>(
    mut commands: Commands,
    generate_tasks: Query<(Entity, &mut GenerateTask<V, N>)>,
) {
    for (id, mut task) in generate_tasks {
        if let Some(volume) = check_ready(task.task_mut()) {
            commands
                .entity(id)
                .insert(volume)
                .remove::<GenerateTask<V, N>>();
        }
    }
}

/// If any async [`MeshTask`]s have been completed, this system will collect the generated [`Mesh`]
/// and add it to the associated entity. If the task was for a subdivide operation, it will wait
/// until all children are ready, and then remove the parent mesh and show the child meshes. If the
/// task was for a merge operation, it will despawn all children and their meshes, and show the
/// parent mesh.
fn poll_mesh_task(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    voxel_material: Res<VoxelMaterial>,
    mut mesh_tasks: Query<(Entity, &mut MeshTask, Option<&Disabled>, Option<&ChildOf>)>,
    children: Query<&Children>,
) {
    let mut not_finished = HashSet::new();

    for (id, mut task, disabled, _) in mesh_tasks.iter_mut() {
        if let Some(Ok(mesh)) = check_ready(task.task_mut()) {
            commands
                .entity(id)
                .insert((
                    Mesh3d(meshes.add(mesh)),
                    MeshMaterial3d(voxel_material.0.clone()),
                ))
                .remove::<MeshTask>();

            // If no `Disabled`, then this is a merge operation and we should despawn children
            if disabled.is_none() {
                commands.entity(id).despawn_children();
            }
        } else {
            not_finished.insert(id);
        }
    }

    // If `Disabled`, this is a subdivide operation. If all subdividing children are ready, remove
    // the mesh from the parent and re-enable the children
    for (_, _, disabled, child_of) in mesh_tasks {
        if disabled.is_some()
            && let Some(parent) = child_of.map(|child_of| child_of.parent())
            && let Ok(children) = children.get(parent)
            && !children.iter().any(|child| not_finished.contains(&child)) {
            for child in children {
                commands.entity(*child).remove::<Disabled>();
            }
            commands
                .entity(parent)
                .remove::<Mesh3d>()
                .remove::<MeshMaterial3d<StandardMaterial>>();
        }
    }
}

/// Subdivides volumes if they are too close, merges volumes if they are too far, and ensures any
/// correctly detailed volumes get mesh tasks if they do not already have a mesh.
#[allow(clippy::type_complexity)]
fn resample_volumes<V: Voxel, const N: usize>(
    mut commands: Commands,
    leaves: Query<(Entity, &Volume<V, N>, &GlobalTransform, Option<&ChildOf>), Without<Children>>,
    branches: Query<(Entity, &Volume<V, N>, &GlobalTransform), (With<Children>, Without<MeshTask>)>,
    viewers: Query<(&Viewer, &GlobalTransform)>,
    meshed: Query<(), With<Mesh3d>>,
) {
    let mut merging = HashSet::new();

    for (id, volume, transform, parent) in leaves {
        // When too close, a volume should subdivide
        if volume.should_upsample(transform, viewers.iter()) {
            commands.entity(id).with_children(|spawner| {
                for bundle in volume.child_tasks(transform) {
                    spawner.spawn(bundle);
                }
            });
        }
        // If there is a parent volume, and it has not already been marked for merging...
        else if let Some((parent_id, parent_volume, parent_transform)) = parent
            .filter(|parent| !merging.contains(&parent.parent()))
            .and_then(|parent| branches.get(parent.parent()).ok())
        {
            // If the parent is too far away, it should merge, meaning it needs a mesh for itself
            if !volume.should_upsample(parent_transform, viewers.iter()) {
                // Adding a mesh task without a disabling component means a merge operation
                commands.entity(parent_id).insert(parent_volume.mesh());
                merging.insert(parent_id);
            }
            // Otherwise, if this child does not have a mesh, start a mesh task
            else if meshed.get(id).is_err() {
                // Adding a mesh task with a disabling component means a subdivision operation
                commands.entity(id).insert((volume.mesh(), Disabled));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{boxy_placer, StandardVolume, StandardVoxel};
    use bevy::{
        core_pipeline::CorePipelinePlugin, mesh::MeshPlugin, pbr::PbrPlugin, render::RenderPlugin,
    };

    fn add_required_plugins(app: &mut App) {
        app.add_plugins((
            MinimalPlugins,
            AssetPlugin::default(),
            ImagePlugin::default(),
            RenderPlugin::default(),
            MeshPlugin,
            CorePipelinePlugin,
            PbrPlugin::default(),
        ));
    }

    #[test]
    fn test_viewer_new() {
        let viewer = Viewer::new(10.0);
        assert_eq!(viewer.lod_distance, 10.0);
    }

    #[test]
    fn test_viewer_should_upsample_close() {
        let viewer = Viewer::new(10.0);
        let viewer_transform = GlobalTransform::from_translation(Vec3::new(0.0, 0.0, 0.0));
        let volume_transform = GlobalTransform::from_translation(Vec3::new(5.0, 0.0, 0.0));
        assert!(viewer.should_upsample(&viewer_transform, &volume_transform));
    }

    #[test]
    fn test_viewer_should_upsample_far() {
        let viewer = Viewer::new(10.0);
        let viewer_transform = GlobalTransform::from_translation(Vec3::new(0.0, 0.0, 0.0));
        let volume_transform = GlobalTransform::from_translation(Vec3::new(100.0, 0.0, 0.0));
        assert!(!viewer.should_upsample(&viewer_transform, &volume_transform));
    }

    #[test]
    fn test_viewer_should_upsample_at_boundary() {
        let viewer = Viewer::new(10.0);
        let viewer_transform = GlobalTransform::from_translation(Vec3::new(0.0, 0.0, 0.0));
        let volume_transform = GlobalTransform::from_translation(Vec3::new(10.0, 0.0, 0.0));
        assert!(viewer.should_upsample(&viewer_transform, &volume_transform));
    }

    #[test]
    fn test_polling_generate() {
        let mut app = App::new();
        add_required_plugins(&mut app);
        app.add_plugins(BevoxPlugin::for_parameters::<StandardVoxel, 16>());
        app.world_mut().run_schedule(Startup);

        fn sampler(_position: Vec3) -> StandardVoxel {
            StandardVoxel::default()
        }

        app.world_mut().spawn(StandardVolume::generate(
            sampler,
            boxy_placer,
            GlobalTransform::default(),
            1,
            false,
        ));

        loop {
            app.world_mut().run_schedule(Update);

            // Check if there exists a newly generated volume
            let mut query_state = app.world_mut().query::<&StandardVolume<StandardVoxel>>();
            if query_state.iter(app.world()).next().is_some() {
                return;
            }
        }
    }

    // TODO: test polling incomplete and complete `MeshTask` for both subdivide and merge
    // operations; test subdividing, merging, mesh task, and no-op in `resample_volumes` for
    // various viewer setups.
}

