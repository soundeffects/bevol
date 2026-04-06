/*! # Meshing Module
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

/// A task handle for an asynchronously generated [`Mesh`]. When polled, the task will
/// complete the mesh generation and return a Bevy [`Mesh`]. The mesh is produced using a
/// dual-contouring algorithm, with vertex placement determined by a function passed in by the
/// user when creating a [`Volume`] using [`Volume::generate`].
#[derive(Component)]
pub(crate) struct MeshTask(Task<Result<Mesh, MeshTaskError>>);

impl MeshTask {
    pub(crate) fn new(
        async_function: impl Future<Output = Result<Mesh, MeshTaskError>> + Send + 'static,
    ) -> Self {
        Self(AsyncComputeTaskPool::get().spawn(async_function))
    }

    pub(crate) fn task_mut(&mut self) -> &mut Task<Result<Mesh, MeshTaskError>> {
        &mut self.0
    }
}

#[derive(Clone)]
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

    fn contains_surface(&self) -> bool {
        self.fields()
            .iter()
            .any(|voxel| voxel.opaque() != self.nx_ny_nz.opaque())
    }
}

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
