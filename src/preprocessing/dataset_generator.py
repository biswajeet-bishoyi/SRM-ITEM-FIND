"""
Dataset Generator for SRM CampusFind.
Generates realistic SRM KTR campus lost/found dataset, ground-truth match pairs,
metadata, and synthetic paraphrased query variations for AI evaluation and model training.
Expanded with 50 diverse campus item pairs (100 item reports).
"""

import json
import os
import random
from datetime import datetime, timedelta

DATASET_PATH = "data/processed/srm_campusfind_dataset.json"
SYNTHETIC_PATH = "data/synthetic/paraphrased_variations.json"

RAW_SEED_ITEMS = [
    # 1 - 8: Core Campus Anchors
    {
        "pair_id": "PAIR_001",
        "category": "electronics",
        "brand": "Boat",
        "color": "black",
        "lost_desc": "I misplaced my black wireless Airdopes earbuds near the central library floor 2.",
        "found_desc": "Found a pair of dark-colored Bluetooth earphones close to the central library study area.",
        "location_lost": "Central Library, 2nd Floor",
        "location_found": "Central Library, Ground Floor Staircase",
        "lost_coords": [12.8231, 80.0442],
        "found_coords": [12.8234, 80.0445],
        "time_lost": "2026-10-04T10:30:00",
        "time_found": "2026-10-04T11:15:00",
        "image_label": "black_earbuds.jpg"
    },
    {
        "pair_id": "PAIR_002",
        "category": "bags",
        "brand": "Nike",
        "color": "navy blue",
        "lost_desc": "Lost my dark blue Nike backpack with laptop inside near Tech Park Java Canteen.",
        "found_desc": "Found a navy blue Nike bag sitting on a table near Java Canteen Tech Park.",
        "location_lost": "Tech Park Java Canteen",
        "location_found": "Java Canteen Seating Area",
        "lost_coords": [12.8245, 80.0455],
        "found_coords": [12.8247, 80.0458],
        "time_lost": "2026-10-04T13:00:00",
        "time_found": "2026-10-04T14:10:00",
        "image_label": "navy_nike_backpack.jpg"
    },
    {
        "pair_id": "PAIR_003",
        "category": "documents",
        "brand": "SRM Institute",
        "color": "blue & white",
        "lost_desc": "Dropped my student ID card with blue lanyard around UB building food court.",
        "found_desc": "Pick up an SRM identity card attached to a blue neck strap outside University Building.",
        "location_lost": "UB Building Food Court",
        "location_found": "UB Ground Floor Lobby",
        "lost_coords": [12.8220, 80.0430],
        "found_coords": [12.8222, 80.0432],
        "time_lost": "2026-10-03T16:45:00",
        "time_found": "2026-10-03T17:20:00",
        "image_label": "srm_id_card.jpg"
    },
    {
        "pair_id": "PAIR_004",
        "category": "electronics",
        "brand": "Apple",
        "color": "space grey",
        "lost_desc": "Misplaced an Apple Watch Series 7 with black silicone strap near sports complex tennis court.",
        "found_desc": "Found a smartwatch with black rubber strap near the outdoor sports ground.",
        "location_lost": "Sports Complex Tennis Court",
        "location_found": "Outdoor Basketball Court Pavilion",
        "lost_coords": [12.8260, 80.0470],
        "found_coords": [12.8263, 80.0474],
        "time_lost": "2026-10-05T08:00:00",
        "time_found": "2026-10-05T09:30:00",
        "image_label": "apple_watch.jpg"
    },
    {
        "pair_id": "PAIR_005",
        "category": "accessories",
        "brand": "Ray-Ban",
        "color": "black",
        "lost_desc": "Lost black frame sunglasses in a brown leather case at MTP Canteen.",
        "found_desc": "Found a pair of black spectacles/sunglasses inside a brown case on MTP table.",
        "location_lost": "MTP Canteen",
        "location_found": "MTP Canteen counter 3",
        "lost_coords": [12.8210, 80.0415],
        "found_coords": [12.8211, 80.0416],
        "time_lost": "2026-10-05T12:15:00",
        "time_found": "2026-10-05T12:40:00",
        "image_label": "rayban_sunglasses.jpg"
    },
    {
        "pair_id": "PAIR_006",
        "category": "accessories",
        "brand": "Milton",
        "color": "silver",
        "lost_desc": "Left my silver stainless steel Milton water bottle in Architecture Block Lab 3.",
        "found_desc": "A metallic steel water bottle was handed over from Architecture department 2nd floor.",
        "location_lost": "Architecture Block Lab 3",
        "location_found": "Architecture Department Office",
        "lost_coords": [12.8251, 80.0421],
        "found_coords": [12.8253, 80.0423],
        "time_lost": "2026-10-02T11:00:00",
        "time_found": "2026-10-02T13:00:00",
        "image_label": "silver_bottle.jpg"
    },
    {
        "pair_id": "PAIR_007",
        "category": "electronics",
        "brand": "Dell",
        "color": "black",
        "lost_desc": "Lost a black 65W Dell laptop charger adapter at Tech Park 6th floor lab.",
        "found_desc": "Found a Dell laptop power brick charger cord left in Tech Park Lab 602.",
        "location_lost": "Tech Park 6th Floor Lab",
        "location_found": "Tech Park Lab 602",
        "lost_coords": [12.8246, 80.0456],
        "found_coords": [12.8246, 80.0457],
        "time_lost": "2026-10-04T15:00:00",
        "time_found": "2026-10-04T16:30:00",
        "image_label": "dell_charger.jpg"
    },
    {
        "pair_id": "PAIR_008",
        "category": "accessories",
        "brand": "Honda",
        "color": "silver & red",
        "lost_desc": "Lost bike keys with a red Honda keychain near Abode Valley gate.",
        "found_desc": "Found two motorcycle keys with red tag near main road outside Abode.",
        "location_lost": "Abode Valley Gate",
        "location_found": "Abode Valley Main Entrance",
        "lost_coords": [12.8205, 80.0390],
        "found_coords": [12.8207, 80.0392],
        "time_lost": "2026-10-05T19:00:00",
        "time_found": "2026-10-05T20:15:00",
        "image_label": "honda_keys.jpg"
    },

    # 9 - 15: Audio, Earbuds & Wearables
    {
        "pair_id": "PAIR_009",
        "category": "electronics",
        "brand": "Apple",
        "color": "white",
        "lost_desc": "Misplaced white AirPods Pro with a clear silicone protective case at TP Ganesan Auditorium.",
        "found_desc": "Discovered Apple wireless earphones inside a transparent case near auditorium entrance.",
        "location_lost": "TP Ganesan Auditorium Main Hall",
        "location_found": "TP Ganesan Auditorium Entrance Stairs",
        "lost_coords": [12.8228, 80.0410],
        "found_coords": [12.8229, 80.0411],
        "time_lost": "2026-10-06T11:00:00",
        "time_found": "2026-10-06T12:30:00",
        "image_label": "airpods_pro.jpg"
    },
    {
        "pair_id": "PAIR_010",
        "category": "electronics",
        "brand": "OnePlus",
        "color": "black",
        "lost_desc": "Lost my black OnePlus Nord Buds case inside University Building central elevator.",
        "found_desc": "Found a matte black OnePlus earphone charging case near UB ground elevator door.",
        "location_lost": "University Building Central Lift",
        "location_found": "UB Ground Floor Elevator Lobby",
        "lost_coords": [12.8220, 80.0430],
        "found_coords": [12.8221, 80.0431],
        "time_lost": "2026-10-06T14:20:00",
        "time_found": "2026-10-06T15:00:00",
        "image_label": "oneplus_buds.jpg"
    },
    {
        "pair_id": "PAIR_011",
        "category": "electronics",
        "brand": "Sony",
        "color": "silver",
        "lost_desc": "Left silver Sony WH-1000XM4 over-ear headphones on study carrel in Central Library 1st floor.",
        "found_desc": "Found large silver-grey Sony noise-canceling headphones in the library reading room.",
        "location_lost": "Central Library 1st Floor Study Carrel",
        "location_found": "Central Library Reading Hall A",
        "lost_coords": [12.8231, 80.0442],
        "found_coords": [12.8232, 80.0443],
        "time_lost": "2026-10-07T16:00:00",
        "time_found": "2026-10-07T17:15:00",
        "image_label": "sony_headphones.jpg"
    },
    {
        "pair_id": "PAIR_012",
        "category": "electronics",
        "brand": "Noise",
        "color": "rose gold",
        "lost_desc": "Misplaced a Noise ColorFit smart watch with rose gold dial and pink strap at Kaveri hostel gym.",
        "found_desc": "Found a ladies smartwatch with metallic rose gold case and pink rubber band near treadmill.",
        "location_lost": "Kaveri Hostel Gym",
        "location_found": "Kaveri Hostel Fitness Room",
        "lost_coords": [12.8270, 80.0480],
        "found_coords": [12.8271, 80.0481],
        "time_lost": "2026-10-07T07:30:00",
        "time_found": "2026-10-07T08:45:00",
        "image_label": "noise_watch.jpg"
    },
    {
        "pair_id": "PAIR_013",
        "category": "electronics",
        "brand": "JBL",
        "color": "blue",
        "lost_desc": "Lost blue JBL Wave earbuds in pocket charging case while walking down BioTech corridor.",
        "found_desc": "Picked up a small blue JBL Bluetooth earphone pod box in BioTech block ground passage.",
        "location_lost": "BioTech Block 2nd Floor Corridor",
        "location_found": "BioTech Block Ground Passage",
        "lost_coords": [12.8238, 80.0460],
        "found_coords": [12.8239, 80.0462],
        "time_lost": "2026-10-08T10:15:00",
        "time_found": "2026-10-08T11:00:00",
        "image_label": "jbl_earbuds.jpg"
    },
    {
        "pair_id": "PAIR_014",
        "category": "electronics",
        "brand": "Fastrack",
        "color": "black",
        "lost_desc": "Dropped black Fastrack smart band watch on Sports Complex outdoor badminton court.",
        "found_desc": "Found a black Fastrack fitness tracker band on the sports complex court boundary.",
        "location_lost": "Sports Complex Badminton Court",
        "location_found": "Sports Complex Court Boundary",
        "lost_coords": [12.8260, 80.0470],
        "found_coords": [12.8262, 80.0472],
        "time_lost": "2026-10-08T18:00:00",
        "time_found": "2026-10-08T19:20:00",
        "image_label": "fastrack_band.jpg"
    },
    {
        "pair_id": "PAIR_015",
        "category": "electronics",
        "brand": "Realme",
        "color": "white",
        "lost_desc": "Lost white Realme Buds Air wireless pods at Clock Tower campus bus stop waiting bench.",
        "found_desc": "Found pebble-shaped white Realme earbud charging case on bench near Clock Tower.",
        "location_lost": "Clock Tower Campus Bus Stop",
        "location_found": "Clock Tower Main Gate Bench",
        "lost_coords": [12.8200, 80.0450],
        "found_coords": [12.8202, 80.0451],
        "time_lost": "2026-10-09T08:30:00",
        "time_found": "2026-10-09T09:10:00",
        "image_label": "realme_buds.jpg"
    },

    # 16 - 24: Computing & Mobile Devices
    {
        "pair_id": "PAIR_016",
        "category": "electronics",
        "brand": "Apple",
        "color": "dark blue",
        "lost_desc": "Lost dark blue Midnight Apple MacBook Air M2 laptop in dark grey felt sleeve in library.",
        "found_desc": "Found an Apple MacBook Air inside a grey felt sleeve on Central Library 3rd floor desk.",
        "location_lost": "Central Library 3rd Floor Digital Lab",
        "location_found": "Central Library 3rd Floor Quiet Zone",
        "lost_coords": [12.8231, 80.0442],
        "found_coords": [12.8232, 80.0444],
        "time_lost": "2026-10-09T14:00:00",
        "time_found": "2026-10-09T15:30:00",
        "image_label": "macbook_air.jpg"
    },
    {
        "pair_id": "PAIR_017",
        "category": "electronics",
        "brand": "HP",
        "color": "black",
        "lost_desc": "Left my black HP Pavilion gaming laptop with green keyboard letters at Tech Park 4th floor.",
        "found_desc": "Found a black HP gaming notebook with green backlit keys in Tech Park Lab 404.",
        "location_lost": "Tech Park 4th Floor Lab 404",
        "location_found": "Tech Park Lab 404 Desk 15",
        "lost_coords": [12.8246, 80.0456],
        "found_coords": [12.8246, 80.0456],
        "time_lost": "2026-10-09T16:30:00",
        "time_found": "2026-10-09T17:45:00",
        "image_label": "hp_pavilion.jpg"
    },
    {
        "pair_id": "PAIR_018",
        "category": "electronics",
        "brand": "Lenovo",
        "color": "platinum grey",
        "lost_desc": "Misplaced a grey Lenovo IdeaPad Slim 3 laptop in UB 5th floor seminar room.",
        "found_desc": "Found a slim grey Lenovo laptop on the podium desk of University Building hall 501.",
        "location_lost": "UB 5th Floor Seminar Hall 501",
        "location_found": "UB 5th Floor Hall 501 Podium",
        "lost_coords": [12.8220, 80.0430],
        "found_coords": [12.8221, 80.0431],
        "time_lost": "2026-10-10T11:45:00",
        "time_found": "2026-10-10T13:00:00",
        "image_label": "lenovo_ideapad.jpg"
    },
    {
        "pair_id": "PAIR_019",
        "category": "electronics",
        "brand": "Apple",
        "color": "space grey",
        "lost_desc": "Lost an Apple iPad 9th Generation tablet with black magnetic flip cover in Architecture studio.",
        "found_desc": "Found an Apple iPad inside a dark folding magnetic folio cover in Architecture drafting room.",
        "location_lost": "Architecture Block Drawing Studio 2",
        "location_found": "Architecture Studio 2 Drafting Table",
        "lost_coords": [12.8251, 80.0421],
        "found_coords": [12.8252, 80.0422],
        "time_lost": "2026-10-10T14:15:00",
        "time_found": "2026-10-10T15:30:00",
        "image_label": "ipad_tablet.jpg"
    },
    {
        "pair_id": "PAIR_020",
        "category": "electronics",
        "brand": "Samsung",
        "color": "black",
        "lost_desc": "Misplaced Samsung Galaxy S23 smartphone in transparent back cover on table at Java Canteen.",
        "found_desc": "Found a black Samsung flagship Android phone with clear cover on Java Canteen corner table.",
        "location_lost": "Tech Park Java Canteen Table 12",
        "location_found": "Java Canteen Lawn Table",
        "lost_coords": [12.8245, 80.0455],
        "found_coords": [12.8247, 80.0457],
        "time_lost": "2026-10-11T13:10:00",
        "time_found": "2026-10-11T13:40:00",
        "image_label": "samsung_s23.jpg"
    },
    {
        "pair_id": "PAIR_021",
        "category": "electronics",
        "brand": "Apple",
        "color": "white",
        "lost_desc": "Dropped white iPhone 14 with cracked glass protector on Open Air Theatre amphitheatre steps.",
        "found_desc": "Found an Apple iPhone in white with damaged screen film sitting on concrete OAT step.",
        "location_lost": "Open Air Theatre Steps",
        "location_found": "OAT Amphitheatre Central Row",
        "lost_coords": [12.8240, 80.0435],
        "found_coords": [12.8241, 80.0436],
        "time_lost": "2026-10-11T19:30:00",
        "time_found": "2026-10-11T20:20:00",
        "image_label": "iphone_14.jpg"
    },
    {
        "pair_id": "PAIR_022",
        "category": "electronics",
        "brand": "OnePlus",
        "color": "silver",
        "lost_desc": "Lost silver OnePlus 11R phone with carbon fiber textured bumper case near MTP juice counter.",
        "found_desc": "Found a silver OnePlus Android mobile in dark carbon pattern case on MTP canteen shelf.",
        "location_lost": "MTP Canteen Juice Corner",
        "location_found": "MTP Canteen Self Service Counter",
        "lost_coords": [12.8210, 80.0415],
        "found_coords": [12.8211, 80.0416],
        "time_lost": "2026-10-12T12:00:00",
        "time_found": "2026-10-12T12:45:00",
        "image_label": "oneplus_11r.jpg"
    },
    {
        "pair_id": "PAIR_023",
        "category": "electronics",
        "brand": "Redmi",
        "color": "blue",
        "lost_desc": "Misplaced blue Redmi Note 12 mobile phone with anime sticker on back in Estates admin block.",
        "found_desc": "Found a light blue Xiaomi Redmi smartphone with cartoon sticker at Estates office lobby.",
        "location_lost": "Estates Block Admin Desk",
        "location_found": "Estates Block Lobby Waiting Area",
        "lost_coords": [12.8202, 80.0448],
        "found_coords": [12.8203, 80.0449],
        "time_lost": "2026-10-12T15:30:00",
        "time_found": "2026-10-12T16:15:00",
        "image_label": "redmi_note.jpg"
    },
    {
        "pair_id": "PAIR_024",
        "category": "electronics",
        "brand": "Nothing",
        "color": "white",
        "lost_desc": "Lost transparent back Nothing Phone (1) with glowing LED lights at UB food court.",
        "found_desc": "Found unique transparent glass Nothing Phone 1 left on table at University Building cafeteria.",
        "location_lost": "UB Food Court 1st Floor",
        "location_found": "UB Cafeteria Table 4",
        "lost_coords": [12.8220, 80.0430],
        "found_coords": [12.8222, 80.0432],
        "time_lost": "2026-10-13T13:40:00",
        "time_found": "2026-10-13T14:30:00",
        "image_label": "nothing_phone.jpg"
    },

    # 25 - 30: Tech Accessories, Chargers & Storage
    {
        "pair_id": "PAIR_025",
        "category": "electronics",
        "brand": "Mi",
        "color": "black",
        "lost_desc": "Left black Mi 20000mAh heavy powerbank with dual fast charging ports at Java Canteen.",
        "found_desc": "Found a matte black Xiaomi powerbank battery pack plugged into wall near Java Canteen.",
        "location_lost": "Tech Park Java Canteen Plug Point",
        "location_found": "Java Canteen Wall Socket Area",
        "lost_coords": [12.8245, 80.0455],
        "found_coords": [12.8246, 80.0457],
        "time_lost": "2026-10-13T16:00:00",
        "time_found": "2026-10-13T17:10:00",
        "image_label": "mi_powerbank.jpg"
    },
    {
        "pair_id": "PAIR_026",
        "category": "electronics",
        "brand": "Anker",
        "color": "white",
        "lost_desc": "Lost white Anker 65W GaN fast charger with thick braided Type-C cable at Central Library.",
        "found_desc": "Found a white multi-port Anker laptop charger adapter with braided cord at library desk.",
        "location_lost": "Central Library Charging Kiosk",
        "location_found": "Central Library Study Table 18",
        "lost_coords": [12.8231, 80.0442],
        "found_coords": [12.8233, 80.0443],
        "time_lost": "2026-10-14T11:20:00",
        "time_found": "2026-10-14T12:15:00",
        "image_label": "anker_charger.jpg"
    },
    {
        "pair_id": "PAIR_027",
        "category": "electronics",
        "brand": "SanDisk",
        "color": "silver",
        "lost_desc": "Lost metallic silver SanDisk 64GB dual OTG pendrive in Mechanical Block CAD workstation.",
        "found_desc": "Found a silver swivel USB drive by SanDisk in Mechanical department computer lab.",
        "location_lost": "Mechanical Block CAD Lab Room 204",
        "location_found": "Mechanical Computer Lab Workstation 9",
        "lost_coords": [12.8255, 80.0445],
        "found_coords": [12.8256, 80.0446],
        "time_lost": "2026-10-14T15:00:00",
        "time_found": "2026-10-14T16:00:00",
        "image_label": "sandisk_usb.jpg"
    },
    {
        "pair_id": "PAIR_028",
        "category": "electronics",
        "brand": "Logitech",
        "color": "black",
        "lost_desc": "Misplaced graphite black Logitech MX Master 3S wireless mouse on Tech Park 7th floor.",
        "found_desc": "Found an ergonomic black Logitech Bluetooth mouse left in Tech Park meeting room.",
        "location_lost": "Tech Park 7th Floor Seminar Room",
        "location_found": "Tech Park 7th Floor Conference Table",
        "lost_coords": [12.8246, 80.0456],
        "found_coords": [12.8247, 80.0457],
        "time_lost": "2026-10-15T10:45:00",
        "time_found": "2026-10-15T11:45:00",
        "image_label": "logitech_mouse.jpg"
    },
    {
        "pair_id": "PAIR_029",
        "category": "electronics",
        "brand": "Portronics",
        "color": "grey",
        "lost_desc": "Left grey aluminum Portronics USB-C multiport hub dongle in UB 3rd floor CSE computer lab.",
        "found_desc": "Found a metallic Type-C dongle adapter hub with HDMI port in UB Lab 302.",
        "location_lost": "UB 3rd Floor CSE Lab 302",
        "location_found": "UB Lab 302 Desk 4",
        "lost_coords": [12.8220, 80.0430],
        "found_coords": [12.8221, 80.0431],
        "time_lost": "2026-10-15T14:30:00",
        "time_found": "2026-10-15T15:20:00",
        "image_label": "usb_hub.jpg"
    },
    {
        "pair_id": "PAIR_030",
        "category": "electronics",
        "brand": "Apple",
        "color": "white",
        "lost_desc": "Lost white square Apple 20W USB-C iPhone power adapter brick in Girls Hostel common room.",
        "found_desc": "Found genuine Apple 20W iPhone wall charging plug in Kaveri hostel ground floor hall.",
        "location_lost": "Girls Hostel Common Study Room",
        "location_found": "Kaveri Hostel Ground Floor Hall",
        "lost_coords": [12.8270, 80.0480],
        "found_coords": [12.8272, 80.0482],
        "time_lost": "2026-10-16T09:00:00",
        "time_found": "2026-10-16T10:00:00",
        "image_label": "apple_adapter.jpg"
    },

    # 31 - 38: Bags, Backpacks & Wallets
    {
        "pair_id": "PAIR_031",
        "category": "bags",
        "brand": "Wildcraft",
        "color": "black",
        "lost_desc": "Misplaced black Wildcraft heavy cargo backpack with red branding near Sports Complex bleachers.",
        "found_desc": "Found a black and red Wildcraft school/college backpack on outdoor sports bleachers.",
        "location_lost": "Sports Complex Stadium Bleachers",
        "location_found": "Sports Complex Ground Pavilion",
        "lost_coords": [12.8260, 80.0470],
        "found_coords": [12.8261, 80.0472],
        "time_lost": "2026-10-16T17:15:00",
        "time_found": "2026-10-16T18:30:00",
        "image_label": "wildcraft_bag.jpg"
    },
    {
        "pair_id": "PAIR_032",
        "category": "bags",
        "brand": "American Tourister",
        "color": "navy blue",
        "lost_desc": "Lost navy blue American Tourister laptop bag with neon yellow zipper pullers at Library.",
        "found_desc": "Found a dark blue American Tourister backpack with yellow tags at Central Library baggage shelf.",
        "location_lost": "Central Library Baggage Counter",
        "location_found": "Central Library Locker Row B",
        "lost_coords": [12.8231, 80.0442],
        "found_coords": [12.8233, 80.0443],
        "time_lost": "2026-10-17T11:00:00",
        "time_found": "2026-10-17T12:10:00",
        "image_label": "tourister_bag.jpg"
    },
    {
        "pair_id": "PAIR_033",
        "category": "bags",
        "brand": "Puma",
        "color": "grey",
        "lost_desc": "Left grey Puma sports daypack with white big cat logo at Architecture model-making workshop.",
        "found_desc": "Found a grey Puma backpack sitting on carpenter table in Architecture workshop.",
        "location_lost": "Architecture Block Model Workshop",
        "location_found": "Architecture Workshop Room 102",
        "lost_coords": [12.8251, 80.0421],
        "found_coords": [12.8252, 80.0423],
        "time_lost": "2026-10-17T15:30:00",
        "time_found": "2026-10-17T16:45:00",
        "image_label": "puma_bag.jpg"
    },
    {
        "pair_id": "PAIR_034",
        "category": "bags",
        "brand": "Fastrack",
        "color": "brown",
        "lost_desc": "Lost dark brown leather Fastrack bi-fold wallet containing student ID and ICICI debit card at MTP.",
        "found_desc": "Found a brown men's leather wallet with bank cards and SRM ID on MTP canteen bench.",
        "location_lost": "MTP Canteen Outer Seating",
        "location_found": "MTP Canteen Bench 5",
        "lost_coords": [12.8210, 80.0415],
        "found_coords": [12.8212, 80.0417],
        "time_lost": "2026-10-18T13:00:00",
        "time_found": "2026-10-18T13:40:00",
        "image_label": "brown_wallet.jpg"
    },
    {
        "pair_id": "PAIR_035",
        "category": "bags",
        "brand": "Tommy Hilfiger",
        "color": "black",
        "lost_desc": "Misplaced black leather Tommy Hilfiger wallet with red/white flag logo at Java Canteen pay desk.",
        "found_desc": "Found a branded black designer wallet near the billing counter at Tech Park Java Canteen.",
        "location_lost": "Tech Park Java Canteen Billing Counter",
        "location_found": "Java Canteen Cash Desk",
        "lost_coords": [12.8245, 80.0455],
        "found_coords": [12.8246, 80.0456],
        "time_lost": "2026-10-18T14:15:00",
        "time_found": "2026-10-18T14:50:00",
        "image_label": "tommy_wallet.jpg"
    },
    {
        "pair_id": "PAIR_036",
        "category": "bags",
        "brand": "Baggit",
        "color": "tan",
        "lost_desc": "Left tan brown Baggit ladies shoulder handbag with gold chain strap in UB reception lobby.",
        "found_desc": "Found a tan women's tote purse with golden metallic buckle in University Building entrance.",
        "location_lost": "UB Ground Floor Reception Sofa",
        "location_found": "UB Main Entrance Lobby Desk",
        "lost_coords": [12.8220, 80.0430],
        "found_coords": [12.8221, 80.0432],
        "time_lost": "2026-10-19T10:30:00",
        "time_found": "2026-10-19T11:20:00",
        "image_label": "baggit_purse.jpg"
    },
    {
        "pair_id": "PAIR_037",
        "category": "bags",
        "brand": "Quechua",
        "color": "green",
        "lost_desc": "Lost khaki green Decathlon Quechua 20L backpack near Clock Tower auto rickshaw stand.",
        "found_desc": "Found an olive green Quechua lightweight rucksack near main campus gate bus shelter.",
        "location_lost": "Clock Tower Auto Stand",
        "location_found": "Main Gate Bus Shelter",
        "lost_coords": [12.8200, 80.0450],
        "found_coords": [12.8203, 80.0452],
        "time_lost": "2026-10-19T16:00:00",
        "time_found": "2026-10-19T17:10:00",
        "image_label": "quechua_bag.jpg"
    },
    {
        "pair_id": "PAIR_038",
        "category": "bags",
        "brand": "Nivia",
        "color": "red",
        "lost_desc": "Dropped red Nivia cylindrical gym duffel bag with running shoes inside at Sports Complex.",
        "found_desc": "Found a red sports kitbag with sneakers inside near Sports Complex changing rooms.",
        "location_lost": "Sports Complex Locker Entrance",
        "location_found": "Sports Complex Changing Room Corridor",
        "lost_coords": [12.8260, 80.0470],
        "found_coords": [12.8262, 80.0471],
        "time_lost": "2026-10-20T08:00:00",
        "time_found": "2026-10-20T09:15:00",
        "image_label": "nivia_duffel.jpg"
    },

    # 39 - 44: Student Documents, Cards & Notebooks
    {
        "pair_id": "PAIR_039",
        "category": "documents",
        "brand": "SRM Institute",
        "color": "red & white",
        "lost_desc": "Lost SRM B.Tech Mechanical ID card on bright red lanyard in Automobile Workshop.",
        "found_desc": "Found an engineering student ID badge with red neck ribbon in workshop machine bay.",
        "location_lost": "Mechanical Automobile Workshop",
        "location_found": "Automobile Lab Bay 2",
        "lost_coords": [12.8255, 80.0445],
        "found_coords": [12.8257, 80.0447],
        "time_lost": "2026-10-20T11:30:00",
        "time_found": "2026-10-20T12:45:00",
        "image_label": "mech_id_card.jpg"
    },
    {
        "pair_id": "PAIR_040",
        "category": "documents",
        "brand": "SRM Institute",
        "color": "green & white",
        "lost_desc": "Misplaced SRM Biotechnology student ID card with green lanyard near BioTech central staircase.",
        "found_desc": "Found a student smart card with green ribbon near ground floor stairs of BioTech department.",
        "location_lost": "BioTech Block Central Stairs",
        "location_found": "BioTech Ground Floor Notice Board",
        "lost_coords": [12.8238, 80.0460],
        "found_coords": [12.8239, 80.0461],
        "time_lost": "2026-10-21T10:00:00",
        "time_found": "2026-10-21T10:50:00",
        "image_label": "biotech_id_card.jpg"
    },
    {
        "pair_id": "PAIR_041",
        "category": "documents",
        "brand": "SRM Institute",
        "color": "white",
        "lost_desc": "Lost semester examination hall ticket and fee receipt inside transparent folder on UB 2nd floor.",
        "found_desc": "Found a plastic document sleeve with SRM exam admit card in University Building 2nd floor wing.",
        "location_lost": "UB 2nd Floor Exam Cell",
        "location_found": "UB 2nd Floor Hallway Bench",
        "lost_coords": [12.8220, 80.0430],
        "found_coords": [12.8222, 80.0431],
        "time_lost": "2026-10-21T14:15:00",
        "time_found": "2026-10-21T15:00:00",
        "image_label": "exam_hall_ticket.jpg"
    },
    {
        "pair_id": "PAIR_042",
        "category": "documents",
        "brand": "Architecture Dept",
        "color": "black",
        "lost_desc": "Left black plastic cylindrical drawing sheet roller container in Architecture Department corridor.",
        "found_desc": "Found an expandable black drawing tube container for engineering sheets on 2nd floor corridor.",
        "location_lost": "Architecture Block 2nd Floor Corridor",
        "location_found": "Architecture 2nd Floor Faculty Lounge",
        "lost_coords": [12.8251, 80.0421],
        "found_coords": [12.8253, 80.0423],
        "time_lost": "2026-10-22T12:00:00",
        "time_found": "2026-10-22T13:10:00",
        "image_label": "sheet_tube.jpg"
    },
    {
        "pair_id": "PAIR_043",
        "category": "documents",
        "brand": "CSE Department",
        "color": "blue",
        "lost_desc": "Lost 2nd Year Data Structures spiral bound lab observation notebook in Tech Park Lab 305.",
        "found_desc": "Found a blue spiral-bound computer science lab record journal in Tech Park 3rd floor lab.",
        "location_lost": "Tech Park Lab 305",
        "location_found": "Tech Park Lab 305 Bench 12",
        "lost_coords": [12.8246, 80.0456],
        "found_coords": [12.8246, 80.0457],
        "time_lost": "2026-10-22T16:00:00",
        "time_found": "2026-10-22T17:00:00",
        "image_label": "lab_record.jpg"
    },
    {
        "pair_id": "PAIR_044",
        "category": "documents",
        "brand": "Govt of India",
        "color": "yellow & blue",
        "lost_desc": "Lost driving license card along with Aadhar card in mini plastic pouch at Library baggage counter.",
        "found_desc": "Found a driver's license ID card inside clear zip pouch on Library entrance floor.",
        "location_lost": "Central Library Baggage Desk",
        "location_found": "Central Library Turnstile Entry",
        "lost_coords": [12.8231, 80.0442],
        "found_coords": [12.8232, 80.0443],
        "time_lost": "2026-10-23T11:00:00",
        "time_found": "2026-10-23T11:40:00",
        "image_label": "driving_license.jpg"
    },

    # 45 - 50: Calculators, Keys, Bottles & Essentials
    {
        "pair_id": "PAIR_045",
        "category": "accessories",
        "brand": "Casio",
        "color": "black & white",
        "lost_desc": "Misplaced Casio fx-991EX Classwiz scientific calculator with protective slip-on lid in UB 402.",
        "found_desc": "Found a black and white Casio engineering calculator with sliding cover in UB classroom 402.",
        "location_lost": "UB 4th Floor Classroom 402",
        "location_found": "UB Room 402 Teacher Podium",
        "lost_coords": [12.8220, 80.0430],
        "found_coords": [12.8221, 80.0431],
        "time_lost": "2026-10-23T13:30:00",
        "time_found": "2026-10-23T14:15:00",
        "image_label": "casio_991ex.jpg"
    },
    {
        "pair_id": "PAIR_046",
        "category": "accessories",
        "brand": "Casio",
        "color": "grey",
        "lost_desc": "Lost grey Casio fx-82MS scientific calculator with name Biswa scratched on back in Mechanical block.",
        "found_desc": "Found a classic grey Casio calculator with engraved initials on desk in Mechanical Room 108.",
        "location_lost": "Mechanical Block Room 108",
        "location_found": "Mechanical Department Class 108",
        "lost_coords": [12.8255, 80.0445],
        "found_coords": [12.8256, 80.0446],
        "time_lost": "2026-10-24T10:00:00",
        "time_found": "2026-10-24T11:00:00",
        "image_label": "casio_82ms.jpg"
    },
    {
        "pair_id": "PAIR_047",
        "category": "accessories",
        "brand": "Royal Enfield",
        "color": "silver & brown",
        "lost_desc": "Lost Royal Enfield Classic bike ignition key with brown leather fob in Abode Valley parking.",
        "found_desc": "Found a motorcycle key attached to an embossed leather RE keychain in Abode parking lot.",
        "location_lost": "Abode Valley Two-Wheeler Parking",
        "location_found": "Abode Valley Parking Block C",
        "lost_coords": [12.8205, 80.0390],
        "found_coords": [12.8206, 80.0391],
        "time_lost": "2026-10-24T18:45:00",
        "time_found": "2026-10-24T19:30:00",
        "image_label": "enfield_key.jpg"
    },
    {
        "pair_id": "PAIR_048",
        "category": "accessories",
        "brand": "Yamaha",
        "color": "black & blue",
        "lost_desc": "Lost Yamaha MT-15 bike key with a blue Monster energy tag at Tech Park scooter parking.",
        "found_desc": "Found a bike key on blue fabric tag lying on ground in Tech Park motorcycle lot.",
        "location_lost": "Tech Park Scooter Parking Area",
        "location_found": "Tech Park Bike Stand Slot 45",
        "lost_coords": [12.8245, 80.0455],
        "found_coords": [12.8247, 80.0456],
        "time_lost": "2026-10-25T09:15:00",
        "time_found": "2026-10-25T10:00:00",
        "image_label": "yamaha_key.jpg"
    },
    {
        "pair_id": "PAIR_049",
        "category": "accessories",
        "brand": "Cello",
        "color": "black",
        "lost_desc": "Left black Cello InsuShield 750ml insulated thermos flask on Open Air Theatre amphitheatre steps.",
        "found_desc": "Found a matte black thermal water flask sitting on Open Air Theatre mid row seats.",
        "location_lost": "Open Air Theatre Mid Steps",
        "location_found": "OAT Stage Right Seating",
        "lost_coords": [12.8240, 80.0435],
        "found_coords": [12.8242, 80.0437],
        "time_lost": "2026-10-25T17:00:00",
        "time_found": "2026-10-25T18:15:00",
        "image_label": "cello_flask.jpg"
    },
    {
        "pair_id": "PAIR_050",
        "category": "accessories",
        "brand": "Tupperware",
        "color": "blue",
        "lost_desc": "Lost translucent blue Tupperware 1 liter water bottle on Sports Complex tennis court bleachers.",
        "found_desc": "Found a blue Tupperware plastic drink bottle on sports court boundary fence.",
        "location_lost": "Sports Complex Tennis Bleachers",
        "location_found": "Sports Complex Court Fence",
        "lost_coords": [12.8260, 80.0470],
        "found_coords": [12.8262, 80.0473],
        "time_lost": "2026-10-26T07:45:00",
        "time_found": "2026-10-26T08:50:00",
        "image_label": "tupperware_bottle.jpg"
    }
]

SYNTHETIC_VARIATIONS_LIST = [
    {
        "pair_id": "PAIR_001",
        "original": "I misplaced my black wireless Airdopes earbuds near the central library floor 2.",
        "variations": [
            "My black wireless earbuds went missing somewhere near the library second floor.",
            "Lost a pair of dark Bluetooth earbuds around central library.",
            "Cannot find my black wireless earpods after studying at central library."
        ]
    },
    {
        "pair_id": "PAIR_002",
        "original": "Lost my dark blue Nike backpack with laptop inside near Tech Park Java Canteen.",
        "variations": [
            "My navy Nike bag containing laptop went missing at Java Canteen.",
            "Left behind a blue Nike rucksack near Java Canteen in Tech Park.",
            "Dark blue Nike bag misplaced at Tech Park canteen tables."
        ]
    },
    {
        "pair_id": "PAIR_003",
        "original": "Dropped my student ID card with blue lanyard around UB building food court.",
        "variations": [
            "Lost SRM college ID card attached to blue strap near UB canteen.",
            "My university identification badge with blue ribbon dropped at UB food court.",
            "Misplaced student identity card near University Building dining area."
        ]
    },
    {
        "pair_id": "PAIR_004",
        "original": "Misplaced an Apple Watch Series 7 with black silicone strap near sports complex tennis court.",
        "variations": [
            "Lost my black Apple smartwatch around the tennis ground.",
            "Dropped an iWatch series 7 with rubber strap at sports complex.",
            "Looking for missing black smartwatch left near outdoor sports courts."
        ]
    },
    {
        "pair_id": "PAIR_007",
        "original": "Lost a black 65W Dell laptop charger adapter at Tech Park 6th floor lab.",
        "variations": [
            "Left my black Dell power brick in Tech park lab 602.",
            "Missing Dell laptop adapter cable on 6th floor TP computer room.",
            "Dell Type-C 65 watt charger misplaced in lab 602."
        ]
    },
    {
        "pair_id": "PAIR_016",
        "original": "Lost dark blue Midnight Apple MacBook Air M2 laptop in dark grey felt sleeve in library.",
        "variations": [
            "My M2 MacBook Air in grey cover went missing on library 3rd floor.",
            "Left behind a midnight blue Apple laptop inside a felt sleeve at central library.",
            "Apple Mac Air misplaced on digital library reading desks."
        ]
    },
    {
        "pair_id": "PAIR_045",
        "original": "Misplaced Casio fx-991EX Classwiz scientific calculator with protective slip-on lid in UB 402.",
        "variations": [
            "Lost my Casio 991EX scientific calculator with cover in classroom 402.",
            "Looking for black and white Casio math calculator left in UB hall 402.",
            "Casio Classwiz calculator misplaced on classroom desk in University Building."
        ]
    }
]


def generate_dataset(base_dir: str = ".", force_regenerate: bool = False):
    """Generate and save structured dataset for SRM CampusFind with 50 pairs (100 items)."""
    dataset_file = os.path.join(base_dir, DATASET_PATH)
    synthetic_file = os.path.join(base_dir, SYNTHETIC_PATH)

    # If dataset already exists on disk and regeneration is not forced, load and return it
    if not force_regenerate and os.path.exists(dataset_file):
        try:
            with open(dataset_file, "r", encoding="utf-8") as f:
                records = json.load(f)
                if records and len(records) >= len(RAW_SEED_ITEMS) * 2:
                    return records
        except Exception:
            pass

    records = []
    rec_counter = 100
    
    for item in RAW_SEED_ITEMS:
        lost_id = f"LOST_{rec_counter}"
        found_id = f"FOUND_{rec_counter}"
        rec_counter += 1
        
        # Lost Report Record
        records.append({
            "id": lost_id,
            "status": "LOST",
            "pair_id": item["pair_id"],
            "match_target_id": found_id,
            "description": item["lost_desc"],
            "category": item["category"],
            "brand": item["brand"],
            "color": item["color"],
            "location_name": item["location_lost"],
            "coordinates": item["lost_coords"],
            "timestamp": item["time_lost"],
            "image_filename": item["image_label"],
            "data_source": "REAL_CURATED"
        })
        
        # Found Report Record
        records.append({
            "id": found_id,
            "status": "FOUND",
            "pair_id": item["pair_id"],
            "match_target_id": lost_id,
            "description": item["found_desc"],
            "category": item["category"],
            "brand": item["brand"],
            "color": item["color"],
            "location_name": item["location_found"],
            "coordinates": item["found_coords"],
            "timestamp": item["time_found"],
            "image_filename": item["image_label"],
            "data_source": "REAL_CURATED"
        })
        
    try:
        os.makedirs(os.path.join(base_dir, "data", "processed"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "data", "synthetic"), exist_ok=True)
        with open(dataset_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
            
        with open(synthetic_file, "w", encoding="utf-8") as f:
            json.dump(SYNTHETIC_VARIATIONS_LIST, f, indent=2)
            
        print(f"[Dataset Generator] Successfully generated {len(records)} records across {len(RAW_SEED_ITEMS)} pairs in '{dataset_file}'")
        print(f"[Dataset Generator] Created synthetic paraphrased variations in '{synthetic_file}'")
    except OSError as e:
        print(f"[Dataset Generator] Note: Read-only filesystem ({e}), using in-memory dataset.")

    return records


if __name__ == "__main__":
    generate_dataset(force_regenerate=True)
