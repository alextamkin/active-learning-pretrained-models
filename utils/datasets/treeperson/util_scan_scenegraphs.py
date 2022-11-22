import os
import json


# Reads .json file, returns dict.
def json_to_dict(filename):
    with open(filename) as f:
        data = json.load(f)
    return data

'''
Given a JSON dictionary and a particular attribute, returns map of
{all values that attribute takes on -> keys that have that attribute}.
'''
def attribute_value_map(dict, att):
    value_map = {}
    for k in dict.keys():
        if att in dict[k]:
            if dict[k][att] in value_map:
                value_map[dict[k][att]].append(k)
            else:
                value_map[dict[k][att]] = [k]
    return value_map

'''
Given a dictionary of imageIDs to GQA Scene Graphs, returns a dictionary
mapping the name of any object found in those images to a set of imageIDs which
contain that object.
'''
def object_frequency_map(dict):
    freq_map = {}
    for k in dict.keys():
        objects = dict[k]["objects"]
        for o in objects.values():
            o_name = o["name"]
            if o_name in freq_map:
                freq_map[o_name].add(k)
            else:
                freq_map[o_name] = {k}
    return freq_map


'''
A function that filters the items in a dictionary and returns a new filtered dictionary.

Takes as input a dictionary, dict, a function, f, and an optional list, keys_list.
The function f must be one that takes the dictionary as a parameter and returns another
function. This returned function takes in dictionary keys as input and returns a boolean to
indicate if that key should remain in the filtered dictionary (True) or if it should be
excluded (False).
If keys_list is provided, then filtering will be done on those keys and the returned
dictionary will not include any keys not in keys_list.

Example usage: filtered_dict = filter_dictionary(dict, select_img_by_attr("location", "outside"))
'''
def filter_dictionary(dict, f, keys_list=None):
    if not keys_list:
        keys_list = dict.keys()
    else:
        keys_list = filter(lambda k: k in dict, keys_list)
    new_dict = {}
    filtered_keys = filter(f(dict), keys_list)
    for k in filtered_keys:
        new_dict[k] = dict[k]
    return new_dict


'''
A filtering function made to be instantiated and used as an argument to filter_dictionary().

Takes in an object attribute (ex: location, weather) and attribute value, and returns a function, f, that can
be passed to filter_dictionary().
Function f takes in a dictionary as an argument and returns a function, g.
Function g takes in keys of the dictionary passed into f and returns True iff the image
has that attribute and that attribute value.

Example usage: select_img_by_attr("location", "outside")
'''
def select_img_by_attr(att, att_val):
    def f(dict):
        def g(k):
            if (att in dict[k]) and (dict[k][att] == att_val):
                return True
            return False
        return g
    return f


'''
A filtering function made to be instantiated and used as an argument to filter_dictionary().

Takes in a list of object names (ex: beach, apple, hat) and returns a function, f, that can
be passed to filter_dictionary().
Function f takes in a dictionary as an argument and returns a function, g.
Function g takes in keys of the dictionary passed into f and returns True iff the image
represented by that key contained an object whose name was in object_name_list.

Example usage: select_img_by_objects(["beach", "ocean", "river"])
'''
def select_img_by_objects(object_name_list):
    def f(dict):
        def g(k):
            objects = dict[k]["objects"]
            for o in objects.values():
                if o["name"] in object_name_list:
                    return True
            return False
        return g
    return f
