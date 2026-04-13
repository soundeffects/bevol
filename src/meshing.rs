/*! # Meshing Module

Handles mesh generation from voxel data using dual-contouring. The meshing algorithm
operates on cells (voxels corners) and produces triangulated meshes suitable for
Bevy rendering.

The module provides:
- [`Cell`] - A collection of 8 voxels representing cell corners for marching cubes-style algorithms
- [`MeshTask`] - Async mesh generation task handle
- [`mesh_cells`] - Core meshing function that transforms voxel cells into renderable meshes
*/

use super::Voxel;
use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology},
    platform::collections::HashMap,
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task},
};

pub(crate) enum MeshTaskError {
    PoisonedRwLock,
}

//TODO: move MeshTask to volume module
/// A task handle for an asynchronously generated [`Mesh`]. When polled, the task will
/// complete the mesh generation and return a Bevy [`Mesh`]. The mesh is produced using a
/// dual-contouring algorithm, with vertex placement determined by a function passed in by the
/// user when creating a [`Volume`] using [`Volume::generate`].
///
/// Internally wraps a Bevy async task that produces a Result, allowing proper error
/// propagation when voxel data cannot be accessed (e.g., due to poison locks).
#[derive(Component)]
pub(crate) struct MeshTask(Task<Result<Mesh, MeshTaskError>>);

impl MeshTask {
    pub(crate) fn new(
        async_function: impl Future<Output = Result<Mesh, MeshTaskError>> + Send + 'static,
    ) -> Self {
        Self(AsyncComputeTaskPool::get().spawn(async_function))
    }

    pub(crate) fn inner_mut(&mut self) -> &mut Task<Result<Mesh, MeshTaskError>> {
        &mut self.0
    }
}

/// Represents the 8 corner voxels of a cell in the voxel grid, used for dual-contouring.
///
/// The naming follows the pattern: nx/px = negative/positive X, ny/py = negative/positive Y,
/// nz/pz = negative/positive Z. For example, `nx_py_pz` is the corner at (-X, +Y, +Z).
///
/// Cells are the fundamental unit of meshing - each cell can potentially contain a surface
/// if its voxels have mixed opacity (some opaque, some transparent).
///
/// # Example
/// ```
/// # use bevox::{Cell, StandardVoxel, Voxel};
/// let cell = Cell {
///     nx_ny_nz: StandardVoxel::new(0, -1.0),
///     nx_ny_pz: StandardVoxel::new(0, -1.0),
///     nx_py_nz: StandardVoxel::new(0, -1.0),
///     nx_py_pz: StandardVoxel::new(0, -1.0),
///     px_ny_nz: StandardVoxel::new(0, -1.0),
///     px_ny_pz: StandardVoxel::new(0, -1.0),
///     px_py_nz: StandardVoxel::new(0, -1.0),
///     px_py_pz: StandardVoxel::new(0, -1.0),
/// };
/// // All voxels are transparent, so cell has no surface
/// let all_transparent = cell.nx_ny_nz.opaque() == false
///     && cell.nx_ny_pz.opaque() == false
///     && cell.nx_py_nz.opaque() == false
///     && cell.nx_py_pz.opaque() == false
///     && cell.px_ny_nz.opaque() == false
///     && cell.px_ny_pz.opaque() == false
///     && cell.px_py_nz.opaque() == false
///     && cell.px_py_pz.opaque() == false;
/// assert!(all_transparent);
/// ```
#[derive(Clone, Default)]
pub struct Cell<V: Voxel> {
    pub nx_ny_nz: V,
    pub nx_ny_pz: V,
    pub nx_py_nz: V,
    pub nx_py_pz: V,
    pub px_ny_nz: V,
    pub px_ny_pz: V,
    pub px_py_nz: V,
    pub px_py_pz: V,
}

impl<V: Voxel> Cell<V> {
    /// Returns all 8 corner voxels as an array, ordered by the bit pattern of the corner index.
    ///
    /// Index mapping: bit 0 = X (px), bit 1 = Y (py), bit 2 = Z (pz)
    pub(crate) fn fields(&self) -> [V; 8] {
        [
            self.nx_ny_nz,
            self.nx_ny_pz,
            self.nx_py_nz,
            self.nx_py_pz,
            self.px_ny_nz,
            self.px_ny_pz,
            self.px_py_nz,
            self.px_py_pz,
        ]
    }

    /// Returns true if this cell contains a surface, meaning some voxels are opaque
    /// and some are transparent. This is the key check for determining if a cell
    /// contributes to the mesh.
    ///
    /// A cell with uniform opacity (all opaque or all transparent) has no surface
    /// and can be skipped during meshing.
    fn contains_surface(&self) -> bool {
        self.fields()
            .iter()
            .any(|voxel| voxel.opaque() != self.nx_ny_nz.opaque())
    }
}

/// Generates a renderable mesh from voxel cells using dual-contouring.
///
/// This function iterates through cells, filters those containing surfaces (mixed opacity),
/// places vertices using the provided placer function, and builds triangle indices for rendering.
///
/// - `cells`: Iterator of (index, Cell) pairs representing the voxel grid positions
/// - `placer`: Function that determines vertex position within each cell based on corner voxels
/// - `flat`: If true, produces flat-shaded mesh by duplicating vertices per-face; otherwise
///   shares vertices across faces for smooth shading
///
/// Returns a Bevy [`Mesh`] ready for rendering with computed normals.
pub(crate) fn mesh_cells<V: Voxel>(
    cells: impl Iterator<Item = (UVec3, Cell<V>)>,
    placer: fn(Cell<V>) -> Vec3,
    flat: bool,
) -> Mesh {
    let mut positions = Vec::new();
    let mut indices = Vec::new();
    let mut cell_map = HashMap::new();

    for (volume_index, cell) in cells.filter(|(_, cell)| cell.contains_surface()) {
        positions.push(placer(cell.clone()) + volume_index.as_vec3());
        let vertex_index = positions.len() - 1;
        cell_map.insert(volume_index, vertex_index);

        // Helper to add indices in correct order for each face
        let mut mesh_face = |axis_1: UVec3, axis_2: UVec3, voxel_1: V, voxel_2: V| {
            let coord_1 = (volume_index * axis_1).max_element();
            let coord_2 = (volume_index * axis_2).max_element();
            if coord_1 > 0 && coord_2 > 0 && let (Some(v1), Some(v2), Some(v3)) = (
                    cell_map.get(&(volume_index - axis_1)),
                    cell_map.get(&(volume_index - axis_2)),
                    cell_map.get(&(volume_index - (axis_1 + axis_2)))
                ) {
                    if voxel_1.opaque() && !voxel_2.opaque() {
                        indices.extend_from_slice(&[
                            vertex_index as u32,
                            *v2 as u32,
                            *v1 as u32,
                            *v2 as u32,
                            *v3 as u32,
                            *v1 as u32,
                        ]);
                    } else if !voxel_1.opaque() && voxel_2.opaque() {
                        indices.extend_from_slice(&[
                            vertex_index as u32,
                            *v1 as u32,
                            *v2 as u32,
                            *v2 as u32,
                            *v1 as u32,
                            *v3 as u32,
                        ]);
                    }
                }
        };

        // X-axis face
        mesh_face(UVec3::Y, UVec3::Z, cell.nx_ny_nz, cell.px_ny_nz);

        // Y-axis face
        mesh_face(UVec3::X, UVec3::Z, cell.nx_ny_nz, cell.nx_py_nz);

        // Z-axis face
        mesh_face(UVec3::Y, UVec3::X, cell.nx_ny_nz, cell.nx_ny_pz);
    }

    let mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );

    if flat {
        mesh.with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            indices
                .into_iter()
                .map(|i| *positions.get(i as usize).unwrap())
                .collect::<Vec<_>>(),
        )
    } else {
        mesh.with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
            .with_inserted_indices(Indices::U32(indices))
    }
    .with_computed_normals()
}

#[cfg(test)]
mod tests {
    use super::{Cell, Voxel};
    use crate::{boxy_placer, StandardVoxel};

    fn make_test_cell(has_surface: bool) -> Cell<StandardVoxel> {
        if has_surface {
            Cell {
                nx_ny_nz: StandardVoxel::new(0, 1.0),
                nx_ny_pz: StandardVoxel::new(0, 1.0),
                nx_py_nz: StandardVoxel::new(0, 1.0),
                nx_py_pz: StandardVoxel::new(0, 1.0),
                px_ny_nz: StandardVoxel::new(0, -1.0),
                px_ny_pz: StandardVoxel::new(0, -1.0),
                px_py_nz: StandardVoxel::new(0, -1.0),
                px_py_pz: StandardVoxel::new(0, -1.0),
            }
        } else {
            Cell {
                nx_ny_nz: StandardVoxel::new(0, 1.0),
                nx_ny_pz: StandardVoxel::new(0, 1.0),
                nx_py_nz: StandardVoxel::new(0, 1.0),
                nx_py_pz: StandardVoxel::new(0, 1.0),
                px_ny_nz: StandardVoxel::new(0, 1.0),
                px_ny_pz: StandardVoxel::new(0, 1.0),
                px_py_nz: StandardVoxel::new(0, 1.0),
                px_py_pz: StandardVoxel::new(0, 1.0),
            }
        }
    }

    #[test]
    fn test_cell_contains_surface() {
        let cell_with_surface = make_test_cell(true);
        assert!(cell_with_surface.contains_surface());

        let cell_without_surface = make_test_cell(false);
        assert!(!cell_without_surface.contains_surface());
    }

    #[test]
    fn test_cell_fields() {
        let cell = make_test_cell(false);
        let fields = cell.fields();
        assert_eq!(fields.len(), 8);
    }

    #[test]
    fn test_cell_default() {
        let cell: Cell<StandardVoxel> = Cell::default();
        assert!(!cell.contains_surface());
    }
}
