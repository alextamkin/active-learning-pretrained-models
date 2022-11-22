from util_scan_scenegraphs import *

# Custsomize this function to perform any dataset exploration you want!
def main():
    filename = "train_sceneGraphs.json"
    train_scenes = json_to_dict(filename)

    print("\nLOCATION VALUES")
    loc_values = attribute_value_map(train_scenes, "location")
    for k in loc_values:
        print("--", str(k) + ":", len(loc_values[k]))

    print("\nWEATHER VALUES")
    weather_values = attribute_value_map(train_scenes, "weather")
    for k in weather_values:
        print("--", str(k) + ":", len(weather_values[k]))

    print("\nNUMBER OF BEACH/OCEAN/RIVERS")
    water_images = filter_dictionary(train_scenes, select_img_by_objects(["beach", "ocean", "river"]))
    print("--", len(water_images.keys()))

    print("\nWEATHER PROFILE OF OCEAN IMAGES")
    ocean_images = filter_dictionary(train_scenes, select_img_by_objects(["ocean"]))
    ocean_weather_values = attribute_value_map(ocean_images, "weather")
    for k in ocean_weather_values:
        print("--", str(k) + ":", len(ocean_weather_values[k]))

    print("\nOVERALL OBJECT FREQUENCY")
    overall_objs = object_frequency_map(train_scenes)
    for k in sorted(list(overall_objs.keys())):
        print("--", str(k) + ":", len(overall_objs[k]))
    
    print("\nOCEANS OBJECT FREQUENCY VS OVERALL")
    ocean_objs = object_frequency_map(ocean_images) 
    for k in sorted(list(ocean_objs.keys())):
        print("--", str(k) + ":", len(ocean_objs[k]), "//", len(overall_objs[k]))


# Start here
if __name__ == "__main__":
    main()
