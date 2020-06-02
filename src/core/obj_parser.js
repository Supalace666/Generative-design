'use strict';

const THREE = require('three');
const OBJLoader = require('three-obj-loader');
OBJLoader(THREE);

const loader = new THREE.OBJLoader();

/**
 * Parse & Construct the mesh given the string of obj file
 * @param text {string}
 * @return {Mesh}
 */
function parseModel(text) {
  // parse & construct the mesh
  const group = loader.parse(text);
  const mesh = group.children[0];
  mesh.material = new THREE.MeshBasicMaterial({
    side: THREE.DoubleSide,
  });
  mesh.position.set(0, 0, 0);
  mesh.updateMatrixWorld(true);

  return mesh;
}

module.exports = {loader, parseModel};
