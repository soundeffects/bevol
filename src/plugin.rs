/*! # Plugin Module
*/

use super::{GenerateTask, MeshTask, Volume, Voxel};
use bevy::{
    ecs::entity_disabling::Disabled, platform::collections::HashSet, prelude::*,
    tasks::futures::check_ready,
};
use std::marker::PhantomData;

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

pub struct BevoxPlugin<V: Voxel, const N: usize> {
    phantom_data: PhantomData<V>,
}

pub type StandardBevoxPlugin<V> = BevoxPlugin<V, 16>;

impl<V: Voxel + 'static, const N: usize> Plugin for BevoxPlugin<V, N> {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_voxel_material).add_systems(
            Update,
            (
                poll_generate_task::<V, N>,
                poll_mesh_task,
                resample_volumes::<V, N>,
            ),
        );
    }
}

#[derive(Resource)]
pub struct VoxelMaterial(Handle<StandardMaterial>);

fn setup_voxel_material(mut commands: Commands, mut materials: ResMut<Assets<StandardMaterial>>) {
    commands.insert_resource(VoxelMaterial(materials.add(StandardMaterial::default())));
}

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

        // If there is a parent volume, and it has not already been marked for merging...
        } else if let Some((parent_id, parent_volume, parent_transform)) = parent
            .filter(|parent| !merging.contains(&parent.parent()))
            .and_then(|parent| branches.get(parent.parent()).ok())
        {
            // If the parent is too far away, it should merge, meaning it needs a mesh for itself
            if !volume.should_upsample(parent_transform, viewers.iter()) {
                commands.entity(parent_id).insert(parent_volume.mesh()); // No `Disabled`: merging
                merging.insert(parent_id);
            }
            // Otherwise, if this child does not have a mesh, start a mesh task
            else if meshed.get(id).is_err() {
                commands.entity(id).insert((volume.mesh(), Disabled)); // `Disabled`: subdividing
            }
        }
    }
}
