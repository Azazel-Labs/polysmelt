xof 0303txt 0032
// A comment, and
# another.
template Mesh { <3D82AB44-62DA-11cf-AB39-0020AF71E433> DWORD nVertices; array Vector vertices[nVertices];
 DWORD nFaces; array MeshFace faces[nFaces]; [...] }
Frame Box{FrameTransformMatrix{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1;;}
Mesh Cube <01234567-89AB-CDEF-0123-456789ABCDEF> {
8;0;0;0;,1;0;0;,0;1;0;,1;1;0;,0;0;-1;,1;0;-1;,0;1;-1;,1;1;-1;;
6;4;1,3,2,0;,4;6,7,5,4;,4;4,5,1,0;,4;3,7,6,2;,4;2,6,4,0;,4;5,7,3,1;;
MeshTextureCoords { 8; -1.#IND00;0;, 1;0;, 0;1;, 1;1;, 0;0;, 1;0;, 0;1;, 1;1;, }
  MeshNormals { 6; 0.0;0.0;1.0;, 0.0;0.0;-1.0;, 0.0;-1.0;0.0;, 0.0;1.0;0.0;, -1.0;0.0;0.0;, 1.0;0.0;0.0;,
   6; 4;0,0,0,0;, 4;1,1,1,1;, 4;2,2,2,2;, 4;3,3,3,3;, 4;4,4,4,4;, 4;5,5,5,5;, } // trailing comment
}}
