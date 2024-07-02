
import re
from copy import deepcopy
from time import process_time
import pandas as pd

def evaluate_rule_with_input(rule, input_values):
    env = {}  # Environment for eval, if needed
    conditions = re.split(r'(&&|\|\|)', rule)  # Splitting conditions within the rule
    for condition in conditions:
        for idx, value in enumerate(input_values):
            # Replace placeholder with actual value
            condition = re.sub(f"train_df.iloc\[:, {idx}]", str(value), condition)
        # Evaluate the condition
        try:
            if not eval(condition, {"__builtins__": None}, env):
                return False
        except Exception as e:
            print(f"Error evaluating condition: {e}")
            return False
    return True


def traverse_tree(input_features, current_branch, final_heur, terminating_rules_dict, value_ranges_dict,column_indices_order_list,leaf_nodes_dict,term_outcome, level=0,last_node = None):
 # Maximum depth reached without finding a satisfying rule
    if level >= len(input_features):
        return None

    if current_branch in leaf_nodes_dict.keys(): 
        # Evaluate terminating rules for the current branch, if any
        rules = terminating_rules_dict.get(current_branch, [])
        for rule in rules:
            if evaluate_rule_with_input(rule, input_features):
                # If a terminating rule is satisfied, return the rule
                top_value1 = term_outcome[rule] 
                return top_value1
#         If no terminating rules are satisfied, return the leaf node's value
        (k,) = leaf_nodes_dict[current_branch]
        return k
    
    
    if current_branch == '1':
        current_node = final_heur.get(current_branch, {})
    else:
        current_node = last_node.get(current_branch, {})
    
    feature_index = column_indices_order_list[level]
    last_node = current_node
    feature_value = input_features[feature_index]
    next_branch = None
    
    # Iterate through child branches to find where the feature value fits
    for child_branch in current_node.keys():
        range_lower = value_ranges_dict.get(child_branch)[0]
        range_upper = value_ranges_dict.get(child_branch)[1]
        if range_lower is None:
            range_lower = float('-inf')
        if range_upper is None:
            range_upper = float('inf')    
        if range_lower < feature_value <= range_upper:
            next_branch = child_branch
            break  # Found the matching child branch
    
    if next_branch is None:
        return None  # No matching branch found for the feature value
        
    # If we have terminating rules for this branch, evaluate them
    if terminating_rules_dict.get(next_branch):
        rules = terminating_rules_dict[next_branch]
        for rule in rules:
            if evaluate_rule_with_input(rule, input_features):
                    top_value3 = term_outcome[rule]
                    return top_value3  # Rule satisfied, return corresponding outcome
                
            
    # No terminating rule satisfied, continue traversal if there are more branches
    if next_branch is not None:
        return traverse_tree(input_features, next_branch, final_heur, terminating_rules_dict, value_ranges_dict,column_indices_order_list,leaf_nodes_dict,term_outcome, level + 1,last_node)



def apply_traversal_to_row(row ,'1', final_heur, terminated_rules_dict, value_ranges_dict,column_indices_order_list,leaf_nodes_dict,term_outcome, level = 0):
    # Convert the row to a list of feature values
    if isinstance(row,list)== 0:
        input_features = row.tolist() 
    result = traverse_tree(input_features, '1', final_heur, terminated_rules_dict, value_ranges_dict,column_indices_order_list,leaf_nodes_dict,term_outcome, level)
    return result

x_df_test['predicted_cluster'] = x_df_test.apply(apply_traversal_to_row, axis=1)

