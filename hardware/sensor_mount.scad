// OpenSCAD (кодон гурван хэмжээст загварчлал). Бүх хэмжээ миллиметрээр.
// Х тэнхлэг дэлгэцийн ирмэг дагуу, урд тал −Y, дээш +Z байна.
// Арын урт уян хавчаарт эсгий нааж, урд богино уруулыг дэлгэцийн хүрээнд тулгана.
// Камерыг халхлахгүй хажуу талд байрлуулж, хоёр линзийн төвийн зөрүүг бодитоор хэмжинэ.
// Мэдрэгчийг урд тавцанд линзийг гадагш харуулан, хоёр шураг болон тусгаарлагчаар тогтооно.
// Арын хавтанг нарийн хуванцар бүслүүрээр ороож хоёр завсраар сүвлэнэ. Холбогчийн үзүүр нээлттэй.
screen_thickness = 6;
lining_allowance = 1;
wall = 2.4;
clip_width = 32;
front_lip = 5;
back_lip = 23;
spring_grip = 0.7;
board_width = 18;
board_length = 23;
board_margin = 2;
sensor_width = 22;
sensor_height = 19;
hole_pitch = 15;
hole_diameter = 2.5;
hole_z = 12;
tie_width = 3.2;
gap = screen_thickness + lining_allowance;
tray_width = board_width + 2*board_margin + 2*(tie_width + wall);
eps = 0.02;
$fn = 48;

assert(gap > spring_grip && spring_grip >= 0 && wall >= 1.2);
assert(clip_width >= sensor_width && sensor_width > hole_pitch + hole_diameter);
assert(hole_z > hole_diameter && sensor_height > hole_z + hole_diameter);
assert(board_length > 2*wall && tie_width > 0);

module clip() {
    // Дээд гүүр, урд уруул, арын доод үзүүрийн уян шахалт.
    translate([-clip_width/2, -wall, 0]) cube([clip_width, gap + 2*wall, wall]);
    translate([-clip_width/2, -wall, -front_lip]) cube([clip_width, wall, front_lip + eps]);
    hull() {
        translate([-clip_width/2, gap, 0]) cube([clip_width, wall, wall]);
        translate([-clip_width/2, gap - spring_grip, -back_lip]) cube([clip_width, wall, wall]);
    }
}

module sensor_plate() {
    difference() {
        translate([-sensor_width/2, -wall, wall-eps])
            cube([sensor_width, wall, sensor_height+eps]);
        for (x = [-hole_pitch/2, hole_pitch/2])
            translate([x, eps, wall + hole_z]) rotate([90, 0, 0])
                cylinder(h=wall + 2*eps, d=hole_diameter);
    }
}

module board_tray() {
    tray_length = board_length + 2*board_margin;
    difference() {
        translate([-tray_width/2, gap, 0]) cube([tray_width, tray_length, wall]);
        for (sign = [-1, 1])
            translate([sign*(board_width/2 + board_margin + tie_width/2),
                       gap + tray_length/2, wall/2])
                cube([tie_width, 4.2, wall + 2*eps], center=true);
    }
    // Хоёр нам хашлага хавтангийн хажуугийн гулсалтыг хязгаарлана.
    for (sign = [-1, 1])
        translate([sign*(board_width/2 + board_margin/2 + wall/2),
                   gap + board_margin, wall - eps])
            translate([-wall/2, 0, 0]) cube([wall, board_length/4, wall]);
}

union() {
    clip();
    board_tray();
    sensor_plate();
}
