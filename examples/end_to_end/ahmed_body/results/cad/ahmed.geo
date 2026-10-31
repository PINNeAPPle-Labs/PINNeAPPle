SetFactory("OpenCASCADE");
L = 1.044; W = 0.389; H = 0.288; R = 0.1; c = 0.05; Ls = 0.222; phi = 25.0 * Pi / 180;
// side profile (x, z), drawn in the x-y plane and turned into the x-z plane: front corners rounded with R
Rectangle(1) = {0, 0, 0, L, H, R};
Rectangle(2) = {R, 0, 0, L - R, H, 0};
s() = BooleanUnion{ Surface{1}; Delete; }{ Surface{2}; Delete; };
Rotate {{1, 0, 0}, {0, 0, 0}, Pi / 2} { Surface{s()}; }
e() = Extrude {0, -W, 0} { Surface{s()}; };
Translate {0, W / 2, c} { Volume{e(1)}; }
// plan profile (x, y), front corners rounded with R, extruded over the height
Rectangle(11) = {0, -W / 2, c, L, W, R};
Rectangle(12) = {R, -W / 2, c, L - R, W, 0};
p() = BooleanUnion{ Surface{11}; Delete; }{ Surface{12}; Delete; };
q() = Extrude {0, 0, H} { Surface{p()}; };
b() = BooleanIntersection{ Volume{e(1)}; Delete; }{ Volume{q(1)}; Delete; };
// rear slant: remove everything above the plane through (L - Ls, top) falling at phi towards the rear
Box(100) = {L - Ls, -1, c + H, 2, 2, 1};
Rotate {{0, 1, 0}, {L - Ls, 0, c + H}, phi} { Volume{100}; }
f() = BooleanDifference{ Volume{b()}; Delete; }{ Volume{100}; Delete; };
Mesh.CharacteristicLengthMax = 0.02;
Mesh.CharacteristicLengthFromCurvature = 1;
Mesh.MinimumElementsPerTwoPi = 24;
