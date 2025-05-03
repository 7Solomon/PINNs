from odbAccess import *
import sys
import csv
import os


# --- ODB File Path ---
# Option 1: Get ODB path from command line argument (if running via 'abaqus python')
# Example usage: abaqus python export_script.py path/to/your_simulation.odb
# odb_path = None
# if len(sys.argv) > 1:
#     odb_path = sys.argv[1]
# else:
#     print("Error: ODB file path not provided as a command line argument.")
#     print("Usage: abaqus python export_script.py <path_to_odb_file>")
#     sys.exit(1)

odb_path =r'C:\Users\Johan\Documents\uni\BA\heat\heat.odb'

# --- Output CSV File Path ---
# Replace with the desired path and name for your output CSV file
# It's often convenient to save it in the same directory as the script or ODB
output_directory = os.path.dirname(odb_path) # Saves CSV in the same folder as ODB
csv_filename = 'abaqus_test_heat_data.csv'
csv_path = os.path.join(output_directory, csv_filename) # <<< --- YOU CAN CHANGE THIS ---

# --- Simulation Specifics ---
step_name = 'Step-1' 

# Name of the Part Instance (often needed if your model isn't just a single part)
# Check your model tree in CAE: Assembly -> Instances
# Set to None if you have a very simple model (e.g., single part, no instances)
# or if you want to attempt accessing assembly-level nodes (less common for results)
instance_name = 'PART-1-1'

# Node Set Name (Optional)
# If you only want data from a specific node set, specify its name here.
# Otherwise, set to None to export data for all nodes in the specified instance/assembly.
node_set_name = None # <<< --- Set to 'Set' or leave as None ---

# Field Output Variable for Temperature
temp_variable = 'NT11' # Usually NT11 for nodal temperature

# ==============================================================================
#                                SCRIPT LOGIC
# ==============================================================================

print(f"--- Abaqus Data Export Script ---")
print(f"ODB File:      {odb_path}")
print(f"Output CSV:    {csv_path}")
print(f"Step Name:     {step_name}")
print(f"Instance Name: {instance_name if instance_name else 'Assembly/Default'}")
print(f"Node Set:      {node_set_name if node_set_name else 'All Nodes'}")
print(f"Temp Variable: {temp_variable}")

# --- Check if ODB file exists ---
if not os.path.exists(odb_path):
    print(f"\nError: ODB file not found at '{odb_path}'")
    sys.exit(1)

odb = None # Initialize ODB variable outside the try block

try:
    # --- Open ODB File ---
    print(f"\nOpening ODB file (read-only)...")
    odb = openOdb(path=odb_path, readOnly=True)
    print("ODB opened successfully.")

    # --- Access Step and Frame ---
    if step_name not in odb.steps:
        print(f"\nError: Step '{step_name}' not found in ODB.")
        print(f"Available steps: {list(odb.steps.keys())}")
        sys.exit(1)
    step = odb.steps[step_name]
    # For steady-state, usually use the last frame
    frame = step.frames[-1]
    print(f"Accessed Step: '{step_name}', Frame: {frame.frameId} (Description: {frame.description})")

    # --- Access Temperature Field Output ---
    if temp_variable not in frame.fieldOutputs:
        print(f"\nError: Temperature variable '{temp_variable}' not found in frame {frame.frameId} of step '{step_name}'.")
        print(f"Available field outputs: {list(frame.fieldOutputs.keys())}")
        sys.exit(1)
    temp_field = frame.fieldOutputs[temp_variable]
    print(f"Accessed Field Output: '{temp_variable}'")

    # --- Determine Nodes to Process ---
    nodes_object = None
    region_object = None # <<< ADDED: Store the object for getSubset region

    if instance_name:
        if instance_name in odb.rootAssembly.instances:
            instance_object = odb.rootAssembly.instances[instance_name] # <<< Get instance object
            nodes_object = instance_object.nodes
            region_object = instance_object # <<< Set initial region to the instance
            print(f"Targeting nodes from instance: '{instance_name}'")
        else:
            print(f"\nError: Instance '{instance_name}' not found in the ODB assembly.")
            print(f"Available instances: {list(odb.rootAssembly.instances.keys())}")
            sys.exit(1)
    else:
        # Attempt assembly level nodes (might work for models without explicit instances)
        if hasattr(odb.rootAssembly, 'nodes') and len(odb.rootAssembly.nodes) > 0:
            nodes_object = odb.rootAssembly.nodes
            region_object = odb.rootAssembly # <<< Use assembly as region (less common for field output subset)
            print("Targeting nodes from Root Assembly (no instance specified).")
            print("Warning: Using rootAssembly as region for getSubset might be inefficient or unsupported for some field types.")
        else:
            # Fallback: If no instance name given AND no assembly nodes, try the first instance
            if len(odb.rootAssembly.instances) > 0:
                first_instance_key = list(odb.rootAssembly.instances.keys())[0]
                instance_object = odb.rootAssembly.instances[first_instance_key] # <<< Get instance object
                nodes_object = instance_object.nodes
                region_object = instance_object # <<< Set region to the instance
                print(f"Warning: No instance specified and no assembly nodes found. Falling back to first instance: '{first_instance_key}'.")
            else:
                 print(f"\nError: Cannot find nodes. No instance name specified, no assembly nodes, and no instances found in the assembly.")
                 sys.exit(1)

    # --- Filter by Node Set (if specified) ---
    target_node_labels = None # Store specific labels if using a set or instance nodes
    if node_set_name:
        node_set_found = False
        node_set = None
        # Check instance-level node sets first if instance is specified
        if instance_name and instance_object and node_set_name in instance_object.nodeSets: # <<< Use instance_object
            node_set = instance_object.nodeSets[node_set_name] # <<< Get set object
            nodes_object = node_set.nodes # Directly get nodes from the set
            region_object = node_set #  Set region to the node set object
            print(f"Filtering nodes using instance node set: '{node_set_name}' ({len(nodes_object)} nodes)")
            node_set_found = True
        # Check assembly-level node sets
        elif node_set_name in odb.rootAssembly.nodeSets:
            node_set = odb.rootAssembly.nodeSets[node_set_name] # <<< Get set object
            nodes_object = node_set.nodes # Directly get nodes from the set
            region_object = node_set # Set region to the node set object
            print(f"Filtering nodes using assembly node set: '{node_set_name}' ({len(nodes_object)} nodes)")
            node_set_found = True
        else:
            print(f"\nWarning: Node set '{node_set_name}' not found in instance '{instance_name}' or assembly.")
            print(f"Available instance node sets: {list(odb.rootAssembly.instances[instance_name].nodeSets.keys()) if instance_name and instance_name in odb.rootAssembly.instances else 'N/A'}")
            print(f"Available assembly node sets: {list(odb.rootAssembly.nodeSets.keys())}")
            print("Proceeding with all nodes from the previously determined instance/assembly.")
            # Keep region_object as the instance/assembly if set not found

    if not nodes_object or len(nodes_object) == 0:
        print("\nError: No nodes found for the specified instance/set criteria.")
        sys.exit(1)

    # Store the labels of the nodes we actually want (all from instance or specific from set)
    target_node_labels = set(n.label for n in nodes_object) # Use a set for fast lookups later
    print(f"Target node labels count: {len(target_node_labels)}")
    print(f"Total nodes to process (from nodes_object): {len(nodes_object)}") # Should match target_node_labels count

    # --- Get Temperature Data Subset for Efficiency ---
    # It's faster to get all relevant temperature data at once using the correct region
    try:
        if not region_object:
             print(f"\nError: Region object (Instance or NodeSet) is not defined.")
             sys.exit(1)

        print(f"Attempting getSubset using region: {region_object.name}") # Debug print
        temp_values_subset = temp_field.getSubset(region=region_object).values # <<< CHANGE: Use region_object
        print(f"Retrieved {len(temp_values_subset)} raw temperature values from region.")

        # Create a dictionary for quick lookup, filtering by our target node labels
        temp_data_dict = {}
        for val in temp_values_subset:
            if val.nodeLabel in target_node_labels: # Filter results if region was broader (e.g., instance)
                 temp_data_dict[val.nodeLabel] = val.data

        print(f"Filtered {len(temp_data_dict)} temperature values matching target nodes.")

        if len(temp_data_dict) == 0 and len(target_node_labels) > 0:
             print(f"\nWarning: No temperature data found for the target nodes within the specified region.")
             # Optionally exit or continue depending on desired behavior
             # sys.exit(1)

    except OdbError as e:
         print(f"\nError during getSubset operation: {e}")
         print(f"Region object used: {region_object.name if region_object else 'None'}")
         print("Make sure the field output variable exists for the specified region.")
         sys.exit(1)
    except AttributeError as e:
         print(f"\nError accessing attribute, possibly related to region object: {e}")
         print(f"Region object type: {type(region_object)}")
         sys.exit(1)


    # --- Extract Coordinates and Write to CSV ---
    print(f"\nExtracting coordinates and writing data to '{csv_path}'...")
    count = 0
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['X', 'Y', 'Temperature'])

        # Iterate through the selected nodes (OdbMeshNodeArray)
        for node in nodes_object:
            node_label = node.label
            coords = node.coordinates

            # Look up the temperature using the dictionary
            if node_label in temp_data_dict:
                temperature = temp_data_dict[node_label]
                # Assuming 2D heat plain, take first two coordinates
                if len(coords) >= 2:
                    writer.writerow([coords[0], coords[1], temperature])
                    count += 1
                else:
                    print(f"Warning: Node {node_label} has insufficient coordinates (<2). Skipping.")
            # else: # This case might happen if getSubset failed or returned partial data
            #     print(f"Warning: Temperature data not found in dictionary for node {node_label}. Skipping.")

    print(f"\nSuccessfully wrote data for {count} nodes to '{csv_path}'.")

except OdbError as e:
    print(f"\nAn Abaqus ODB Error occurred: {e}")
except KeyError as e:
    print(f"\nError: A specified key (like step, instance, or field name) was not found: {e}")
except FileNotFoundError as e:
     print(f"\nError: File not found: {e}")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")

finally:
    # --- Close ODB File ---
    if odb:
        odb.close()
        print("ODB file closed.")

print("\n--- Script Finished ---")