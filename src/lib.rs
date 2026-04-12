/*! # Voxel Plugin
Bevy plugin to manage a voxel world based on a sparse octree hierarchy for level-of-detail
rendering.

- [`Volume`] components represent objects or regions of space that are filled in by voxel data
- [`Volume`] entites are positioned in the world with a [`bevy::prelude::Transform`] component
- [`Volume`]s are generic on a [`Voxel`] trait, allowing users to store arbitrary spatial data
- [`Volume`]s recieve a `fn(Vec3) -> Voxel`, used for generation
- [`Volume`]s take a vertex placement function which allows stylization, i.e. blocky or smooth mesh
- Generation and meshing of volumes happen asynchronously.
- Camera entities with the [`Viewer`] component determine the focal points of level-of-detail.

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
- Support for big_space crate
*/

mod meshing;
mod placing;
mod plugin;
mod volume;
mod voxel;

pub use meshing::Cell;
use meshing::{mesh_cells, MeshTask, MeshTaskError};
pub use placing::{boxy_placer, contour_placer, smooth_placer};
pub use plugin::{BevoxPlugin, Viewer};
pub use volume::{GenerateTask, StandardVolume, Volume};
pub use voxel::{StandardVoxel, Voxel};
