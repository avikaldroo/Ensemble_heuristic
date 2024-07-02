import re
from copy import deepcopy
from time import process_time
import math as mt
import pandas as pd
import time
def partition_rule_by_single_value_range(rules_dict, single_range):
    """
    Partition rules based on a specified single value range, accurately allocating counts based on overlaps and bounds,
    and include the value range info in the partitioned counts.
    """
    # Convert the single_range into a string key for easy identification
    range_key = f"{single_range[0]} to {single_range[1]} ({'inclusive' if single_range[2] else 'exclusive'}, {'inclusive' if single_range[3] else 'exclusive'})"
    
    # Initialize counts for the partition with range information
    partition_count = {range_key: rules_dict.get('other_rules', {}).get('total_count', 0)}
    lt = []
    for operator, limits in rules_dict.items():
        if operator == 'other_rules':  # Skip 'other_rules' since it's already counted in initialization
            continue
        
        for limit, count in limits.items():          
            if operator in ['<=', '<','>=', '>']:
                key = "("+ operator+","+str(limit)+")"
                start, end, start_inclusive, end_inclusive = single_range
                
                # Determine inclusion based on operator and range bounds
                in_range = False
                if operator in ['<=', '<']:
                    if (end is not None and start is not None and limit > start ):
                        in_range = True
                    if (end is not None and start is None):
                        in_range = True
                    if (end is None and start is not None and start < limit):
                        in_range = True
                
                if operator in ['>=', '>']:
                    if (start is not None and end is not None and limit < end):
                        in_range = True
                    if (start is None and limit < end):
                        in_range = True
                    if (end is None and start is not None):
                        in_range = True
                 
                if in_range:
                    partition_count[range_key] += count
                    lt.append(key)
            else:
                lower_lt = limit[0] 
                upper_lt = limit[1]
                op = operator.split(' and ')
                key = "("+op[0]+","+ str(lower_lt)+")"+" " +"and"+" "+"("+op[1]+","+str(upper_lt)+")"
                start, end, start_inclusive, end_inclusive = single_range
                in_range = False
                if (start is not None and end is not None and upper_lt > start  and upper_lt <= end):
                    in_range = True
                if (start is not None and end is not None and lower_lt >= start  and lower_lt < end):
                    in_range = True
                if (start is not None and end is not None and lower_lt <= start  and upper_lt >= end):
                    in_range = True
                if (start is None and end is not None and lower_lt <= end):
                    in_range = True
                if (start is not None and end is None and upper_lt > start):
                    in_range = True
                if in_range:
                    partition_count[range_key] += count
                    lt.append(key)
    
    return partition_count, lt
#####################################################################################################################################################
def modify_elements(test_column_index, value_ranges, combined_list):
    # Iterate over each value range tuple only once

    lower_bound = value_ranges[0][0]; upper_bound = value_ranges[0][1]; include_lower = value_ranges[0][2]; include_upper = value_ranges[0][3]

    # Initialize an empty list to hold parts of the condition string
    conditions = []

    # Check and append the appropriate condition based on lower bound
    if lower_bound is not None:
        if include_lower:
            conditions.append(f'(x_df.iloc[:, {test_column_index}] >= {lower_bound})')
        else:
            conditions.append(f'(x_df.iloc[:, {test_column_index}] > {lower_bound})')

    # Check and append the appropriate condition based on upper bound
    if upper_bound is not None:
        if include_upper:
            conditions.append(f'(x_df.iloc[:, {test_column_index}] <= {upper_bound})')
        else:
            conditions.append(f'(x_df.iloc[:, {test_column_index}] < {upper_bound})')

    # Combine conditions into a single string
    condition_str = ' & '.join(conditions)
        
        # Append the condition string to each element in combined_list
    for i in range(len(combined_list)):
            combined_list[i] = combined_list[i] + ' & ' + condition_str if combined_list[i] else condition_str
    return combined_list
#####################################################################################################################################################

def modify_elements_ck(test_column_index, value_ranges, combined_list):
    lt = combined_list.copy()

    lower_bound = value_ranges[0]; upper_bound = value_ranges[1]; include_lower = value_ranges[2]; include_upper = value_ranges[3]
    # Initialize an empty list to hold parts of the condition string
    conditions = []

    # Check and append the appropriate condition based on lower bound
    if lower_bound is not None:
        if include_lower:
            conditions.append(f'(x_df.iloc[:, {test_column_index}] >= {lower_bound})')
        else:
            conditions.append(f'(x_df.iloc[:, {test_column_index}] > {lower_bound})')

    # Check and append the appropriate condition based on upper bound
    if upper_bound is not None:
        if include_upper:
            conditions.append(f'(x_df.iloc[:, {test_column_index}] <= {upper_bound})')
        else:
            conditions.append(f'(x_df.iloc[:, {test_column_index}] < {upper_bound})')

    # Combine conditions into a single string
    condition_str = ' & '.join(conditions)
        
        # Append the condition string to each element in combined_list
    for i in range(len(lt)):
            lt[i] = lt[i] + ' & ' + condition_str if lt[i] else condition_str
    return lt
#####################################################################################################################################################


def categorize_count_collect_others_and_list_rules(rules, column_index):
    pattern = r'\(x_df.iloc\[:, ' + str(column_index) + r'\](<=|>=|<|>)((?:\d+\.?\d*)|(?:\.\d+))\)'

    categories = {
        '>': {}, '<': {}, '>=': {}, '<=': {},
        '> and <': {}, '> and <=': {}, '>= and <': {}, '>= and <=': {},
        'other_rules': {'rules': [], 'total_count': 0}
    }
    rules_by_limit = {}

    for rule in rules:
        matches = re.findall(pattern, rule)
        match_operators = set(match[0] for match in matches)
        match_dict = {match[0]: float(match[1]) if '.' in match[1] else int(match[1]) for match in matches}

        if not matches:
            categories['other_rules']['rules'].append(rule)
            categories['other_rules']['total_count'] += 1
            continue

        if len(matches) > 1:
            process_multiple_conditions_with_list(match_operators, match_dict, categories, rules_by_limit, rule)
        else:
            operator, value = matches[0]
            value = match_dict[operator]
            categories[operator][value] = categories[operator].get(value, 0) + 1
            key = f"({operator},{value})"
            rules_by_limit.setdefault(key, []).append(rule)

    for operator in categories:
        if operator not in ['other_rules']:
            is_ascending = operator in ['>', '>=']
            categories[operator] = dict(sorted(categories[operator].items(), key=lambda item: item[0], reverse=not is_ascending))

    return categories, rules_by_limit
#####################################################################################################################################################
def process_multiple_conditions_with_list(match_operators, match_dict, categories, rules_by_limit, rule):
    combined_conditions = ['> and <', '> and <=', '>= and <', '>= and <=']
    for condition in combined_conditions:
        if all(op in match_operators for op in condition.split(' and ')):
            condition_key = tuple(match_dict[op] for op in condition.split(' and '))
            categories[condition][condition_key] = categories[condition].get(condition_key, 0) + 1
            key = " and ".join(f"({op},{match_dict[op]})" for op in condition.split(' and '))
            rules_by_limit.setdefault(key, []).append(rule)
#####################################################################################################################################################
def generate_value_ranges(lower_limit, upper_limit, interval_length):
    """Generate value ranges based on the given limits and interval length."""
    value_ranges = []
    current_start = lower_limit
    while current_start < upper_limit:
        current_end = min(current_start + interval_length, upper_limit)
        current_start = round(current_start,2)
        current_end = round(current_end,2)
        value_ranges.append((current_start, current_end))
        current_start = current_end
    return value_ranges

def partition_rules(rules, feature_index, x_df, interval_length, lower_limit, upper_limit):
    # Generate initial value ranges for the feature
    value_ranges = generate_value_ranges(lower_limit, upper_limit, interval_length)
    
    partitions = []
    unique_clusters = []
    rule_lt = []
    for start, end in value_ranges:
        # Modify rules for the current partition
        # Note: Implement rule modification based on start and end values for the feature_index
        unique_clusters = []
        vl_range = []
        x = False
        y = True
        in_lt = []
        in_lt.append(start)
        in_lt.append(end)
        in_lt.append(x)
        in_lt.append(y)
        vl_range.append(in_lt)
    

        modified_rules = modify_elements_ck(feature_index, vl_range[0], rules) # Placeholder for actual rule modification logic
        rule_lt.append(modified_rules)
        # Evaluate modified rules on x_df and determine unique cluster values
        # Note: Implement evaluation logic
        cl = set()
        for i in range(len(modified_rules)):   
            ck_df = x_df[eval(modified_rules[i])]
            if ck_df.empty == 0:
                value_counts = ck_df['Cluster'].value_counts()
                top_value = value_counts.index[0] 
                cl.add(top_value) # Placeholder for actual evaluation
        unique_clusters.extend(cl)
        lt = unique_clusters.copy()
        partitions.append((start, end, lt))
    # Club partitions based on the criteria
    final_partitions = []
    i = 0
    while i < len(partitions):
        start, end, cluster_values = partitions[i]
        # Initialize current clubbed range with the first partition's range
        current_start, current_end = start, end
        current_clusters = cluster_values
        while i + 1 < len(partitions) and (
            partitions[i + 1][2] == current_clusters or 
            partitions[i + 1][2] == [] or 
            current_clusters == []
        ):
            next_start, next_end, next_clusters = partitions[i + 1]
            current_end = next_end # Extend the current clubbed range
            if current_clusters == [] or partitions[i + 1][2] == []:
                current_clusters.extend(next_clusters)
            else:
                 current_clusters = next_clusters
            i += 1
        i += 1
        final_partitions.append((current_start, current_end, current_clusters))
        
    
    return final_partitions
#####################################################################################################################################################
def append_to_inner_key(d, target_key, new_key= None, new_value= None,add_value = False):
    """
    Recursively search for a target_key in a nested dictionary and append a new key-value pair
    to its value, assuming the value is also a dictionary.

    :param d: The dictionary to search and update.
    :param target_key: The key whose value should be appended with a new key-value pair.
    :param new_key: The new key to add to the target_key's value.
    :param new_value: The value associated with the new_key.
    """
    for key, value in d.items():
        # If the current key is the target_key and its value is a dictionary, append the new key-value pair
        if add_value == False:
            if key == target_key and isinstance(value, dict):
                value[new_key] = new_value
            elif isinstance(value, dict):
                append_to_inner_key(value, target_key, new_key, new_value)
        else:
            if key == target_key :
                d[key] = new_value
            elif isinstance(value, dict):
                append_to_inner_key(value, target_key, new_key, new_value,add_value = True)
#####################################################################################################################################################
def filter_rules(rules, x_df):
    non_null_rules = []
    for rule in rules:
        try:
            if not x_df[eval(rule)].empty:
                non_null_rules.append(rule)
        except SyntaxError:
            continue  # Skip rules that lead to errors or null sets
    return non_null_rules
#####################################################################################################################################################
def none_leaf(rules,x_df):
    # processing nodes which exhaussted all the features but do not have a unique outcome.
    cluster_indices = {}  # To store unique indices per cluster
    
    for rule in rules:
        filtered_df = x_df[eval(rule)]
        
        # Iterate through each cluster in the filtered DataFrame
        if not filtered_df.empty:
            for cluster in filtered_df['Cluster'].unique():
                cluster_df = filtered_df[filtered_df['Cluster'] == cluster]
                # Collect unique indices for each cluster
                if cluster in cluster_indices:
                    cluster_indices[cluster].update(cluster_df.index)
                else:
                    cluster_indices[cluster] = set(cluster_df.index)
    
    # Calculate distinct counts for each cluster and find the max
    if cluster_indices:
        cluster_max_distinct_count = max(cluster_indices, key=lambda k: len(cluster_indices[k]))
    return cluster_max_distinct_count
#####################################################################################################################################################
def is_leaf_node(rules, x_df,depth, tot_feat):
    unique_clusters = set()
    if depth >= tot_feat:
         unique_clusters.update(set(none_leaf(rules,x_df)))
    elif depth < tot_feat:
        for rule in rules:
            try:
                evaluated_df = x_df[eval(rule)]
                if not evaluated_df.empty:
                    unique_clusters.update(set(evaluated_df['Cluster'].unique()))
            except Exception as e:
                continue  # Skip rules that lead to errors or null sets
    # Return both the leaf node status and the unique clusters
    return len(unique_clusters) <= 1, unique_clusters if len(unique_clusters) == 1 else {None}
#####################################################################################################################################################
def matches_input_indices(s, input_indices):
    # Extract all indices found in the string
    found_indices = {int(x) for x in re.findall(r'iloc\[:, (\d+)]', s)}
    # Check if every found index is within the set of input indices
    return found_indices.issubset(input_indices) and not found_indices.isdisjoint(input_indices)
#####################################################################################################################################################

def generate_value_ranges_fl(partitions):
    value_ranges = []

    for l in range(len(partitions)):
        if l == 0:
            in_lt = (None,partitions[l][1],False,True)
        elif l > 0 and l < (len(partitions)-1):
            in_lt = (partitions[l][0],partitions[l][1],False,True) 
        elif l == (len(partitions)-1):
            in_lt = (partitions[l][0],None,False,False)
        value_ranges.append(in_lt)
    return value_ranges
#####################################################################################################################################################
def build_heuristic(branch, x_df, heur_dict, terminating_rules_dict, value_ranges_dict, column_indices_order_list, partition_parameters_dict, leaf_nodes_dict,final_heur,ex_lt,parent_key= None):
    current_depth = branch.count('_')+1
    if parent_key is None:  # Root node
        if branch not in final_heur:
            final_heur[branch] = {}

    
    if current_depth - 1 >= len(column_indices_order_list):
        _, clusters = is_leaf_node(heur_dict.get(branch, []), x_df,current_depth,len(column_indices_order_list))
        leaf_nodes_dict[branch] = clusters
        append_to_inner_key(final_heur, branch,new_key = None,new_value= clusters,add_value = True)
        return
    
    branch_rules = [item for item in heur_dict.get(branch, []) if item not in terminating_rules_dict.get(branch, []) ] 
    feature_index = column_indices_order_list[current_depth - 1]
    partition_params = partition_parameters_dict[feature_index]
    lower_limit, upper_limit, interval_length = partition_params.values()
    
    filtered_rules = filter_rules(heur_dict.get(branch, []), x_df)
    index_till_now = set()
    for i in range(current_depth):
        index_till_now.add(column_indices_order_list[i])
    
    extended_categorized_counted_conditions, extended_rules_by_limit = categorize_count_collect_others_and_list_rules(filtered_rules, feature_index)
    leaf, clusters = is_leaf_node(filtered_rules, x_df,current_depth,len(column_indices_order_list))
    if leaf or not filtered_rules:
        leaf_nodes_dict[branch] = clusters
        append_to_inner_key(final_heur, branch,new_key = None,new_value= clusters,add_value = True)  
        return
    

    # Placeholder for actual logic to generate partitions and value ranges

    partitions = partition_rules(filtered_rules, feature_index, x_df, interval_length, lower_limit, upper_limit)

    value_ranges = generate_value_ranges_fl(partitions)
    tot_time = 0
    for k, vl_range in enumerate(value_ranges):
        branch_lt = []
        # Process each value range to modify rules accordingly
        _, keys = partition_rule_by_single_value_range(extended_categorized_counted_conditions, vl_range)
        keys_to_iterate = keys
        combined_list1 = [item for key in keys_to_iterate if key in extended_rules_by_limit for item in extended_rules_by_limit[key]]
        combined_list1.extend(extended_categorized_counted_conditions['other_rules']['rules']) 
        combined_list = modify_elements_ck(feature_index, vl_range, combined_list1)

        # Create new branch number and update heur_dict 
        new_branch = f"{branch}_{k + 1}"
        branch_lt.append(new_branch)
        heur_dict[new_branch] = combined_list
        value_ranges_dict[new_branch] = vl_range
        if leaf == 0:
            append_to_inner_key(final_heur, branch,new_key = new_branch,new_value= {})
        filtered_strings = []
        filtered_strings = [s for s in combined_list if  matches_input_indices(s, index_till_now)]
        terminating_rules_dict[new_branch] = filtered_strings
        # Recursive call to process the new branch
        build_heuristic(new_branch, x_df, heur_dict, terminating_rules_dict, value_ranges_dict, column_indices_order_list, partition_parameters_dict, leaf_nodes_dict,final_heur,ex_lt,branch)
################################################################################################################################################################################################

def initiate_heuristic(rules_dict, train_df, column_indices_order_list,branch = '1',parent_key= None,ft_limit_dict = {0: {"lower_limit": 0,"upper_limit":4000,"interval": 50},
1: {"lower_limit": 0,"upper_limit":600,"interval": 10},
2: {"lower_limit": 0,"upper_limit":100,"interval": 5},
3: {"lower_limit": 0,"upper_limit":4100,"interval": 25},
4: {"lower_limit": 0,"upper_limit":200,"interval": 5},
5: {"lower_limit": 0,"upper_limit":1,"interval": 0.05},
6: {"lower_limit": 0,"upper_limit":4,"interval": 0.1},
7: {"lower_limit": 0,"upper_limit":1,"interval": 0.05},
8: {"lower_limit": 0,"upper_limit":1,"interval": 0.05}
} ):
    all_rules_lt = []
    rule_outcome = {}
    term_outcome = {}
    for cluster_value in rules_dict.keys():
        for rule in rules_dict[cluster_value]['rules']:
            all_rules_lt.append(rule)
            rule_outcome[rule] = cluster_value
    heur_dict = {}
    heur_dict["1"] = all_rules_lt
    final_heur = {}
    terminating_rules_dict = {} 
    value_ranges_dict = {} 
    leaf_nodes_dict = {} 
    build_heuristic(branch, train_df, heur_dict, terminating_rules_dict, value_ranges_dict, column_indices_order_list, ft_limit_dict,leaf_nodes_dict,final_heur)
    for x in terminating_rules_dict.keys():
        for y in terminating_rules_dict[x]:
            term_outcome[y] =  rule_outcome[y]

    return final_heur,value_ranges_dict,leaf_nodes_dict,terminating_rules_dict,term_outcome,heur_dict
################################################################################################################################################################################################