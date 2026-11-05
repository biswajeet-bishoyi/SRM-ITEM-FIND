"""
Dataset Generator for SRM CampusFind.
Generates realistic SRM KTR campus lost/found dataset, ground-truth match pairs,
metadata, and synthetic paraphrased query variations for AI evaluation.
"""

import json
import os
import random
from datetime import datetime, timedelta

DATASET_PATH = "data/processed/srm_campusfind_dataset.json"
SYNTHETIC_PATH = "data/synthetic/paraphrased_variations.json"

RAW_SEED_ITEMS = [
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
    }
]

def generate_dataset(base_dir: str = "."):
    """Generate and save structured dataset for SRM CampusFind."""
    os.makedirs(os.path.join(base_dir, "data", "processed"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "data", "synthetic"), exist_ok=True)
    
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
        
    dataset_file = os.path.join(base_dir, DATASET_PATH)
    with open(dataset_file, "w") as f:
        json.dump(records, f, indent=2)
        
    synthetic_file = os.path.join(base_dir, SYNTHETIC_PATH)
    with open(synthetic_file, "w") as f:
        json.dump(SYNTHETIC_VARIATIONS_LIST, f, indent=2)
        
    print(f"[Dataset Generator] Created {len(records)} records in '{dataset_file}'")
    print(f"[Dataset Generator] Created synthetic paraphrased variations in '{synthetic_file}'")
    return records

if __name__ == "__main__":
    generate_dataset()
