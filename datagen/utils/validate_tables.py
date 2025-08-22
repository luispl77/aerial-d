#!/usr/bin/env python3
"""
Dataset Table Validation Script
Performs comprehensive cross-table mathematical verification for AerialSeg thesis tables.
"""

def validate_dataset_statistics():
    """Validate internal consistency of Dataset Statistics Summary table."""
    print("=== Dataset Statistics Summary Validation ===")
    
    # Dataset Statistics Summary data
    stats = {
        'Total Patches': {'train': 27480, 'val': 9808, 'total': 37288},
        'Individual Objects with Expressions': {'train': 94179, 'val': 34536, 'total': 128715},
        'Individual Expressions': {'train': 646686, 'val': 242668, 'total': 889354},
        'Groups with Expressions': {'train': 96832, 'val': 34162, 'total': 130994},
        'Group Expressions': {'train': 471108, 'val': 162061, 'total': 633169},
        'Total Samples': {'train': 191011, 'val': 68698, 'total': 259709},
    }
    
    # Check Train + Val = Total for each metric
    all_good = True
    for metric, values in stats.items():
        calculated_total = values['train'] + values['val']
        if calculated_total != values['total']:
            print(f"❌ {metric}: {values['train']} + {values['val']} = {calculated_total}, but total shows {values['total']}")
            all_good = False
        else:
            print(f"✅ {metric}: {values['train']} + {values['val']} = {values['total']}")
    
    # Check averages
    avg_individual = stats['Individual Expressions']['total'] / stats['Individual Objects with Expressions']['total']
    avg_group = stats['Group Expressions']['total'] / stats['Groups with Expressions']['total']
    
    print(f"\n--- Average Calculations ---")
    print(f"✅ Avg Expressions per Individual Object: {avg_individual:.2f} (should be ~6.91)")
    print(f"✅ Avg Expressions per Group: {avg_group:.2f} (should be ~4.83)")
    
    # Check Total Samples logic
    total_expressions = stats['Individual Expressions']['total'] + stats['Group Expressions']['total']
    print(f"\n--- Total Sample Check ---")
    print(f"Individual + Group Expressions: {stats['Individual Expressions']['total']} + {stats['Group Expressions']['total']} = {total_expressions}")
    print(f"This should relate to Total Samples: {stats['Total Samples']['total']}")
    
    return all_good

def validate_expression_taxonomy():
    """Validate Expression Taxonomy table totals."""
    print("\n=== Expression Taxonomy Validation ===")
    
    # Expression Taxonomy data
    taxonomy = [
        {'combo': 'Category + Position + Relationship', 'individual': 25403, 'group': 86307, 'total': 111710},
        {'combo': 'Category + Position', 'individual': 26437, 'group': 40281, 'total': 66718},
        {'combo': 'Category', 'individual': 5157, 'group': 61015, 'total': 66172},
        {'combo': 'Category + Position + Color', 'individual': 58252, 'group': 0, 'total': 58252},
        {'combo': 'Category + Position + Color + Relationship', 'individual': 42165, 'group': 0, 'total': 42165},
        {'combo': 'Category + Position + Extreme + Color', 'individual': 35571, 'group': 0, 'total': 35571},
        {'combo': 'Category + Extreme + Color', 'individual': 35571, 'group': 0, 'total': 35571},
        {'combo': 'Category + Extreme', 'individual': 22930, 'group': 0, 'total': 22930},
        {'combo': 'Category + Position + Extreme', 'individual': 22930, 'group': 0, 'total': 22930},
        {'combo': 'Category + Color', 'individual': 19172, 'group': 0, 'total': 19172},
        {'combo': 'Category + Position + Extreme + Color + Relationship', 'individual': 15242, 'group': 0, 'total': 15242},
        {'combo': 'Category + Position + Extreme + Relationship', 'individual': 9761, 'group': 0, 'total': 9761},
    ]
    
    # Validate individual row calculations
    all_good = True
    for row in taxonomy:
        calculated_total = row['individual'] + row['group']
        if calculated_total != row['total']:
            print(f"❌ {row['combo']}: {row['individual']} + {row['group']} = {calculated_total}, but total shows {row['total']}")
            all_good = False
        else:
            print(f"✅ {row['combo']}: {row['individual']} + {row['group']} = {row['total']}")
    
    # Calculate totals
    total_individual = sum(row['individual'] for row in taxonomy)
    total_group = sum(row['group'] for row in taxonomy)
    total_all = sum(row['total'] for row in taxonomy)
    
    print(f"\n--- Taxonomy Totals ---")
    print(f"Total Individual Expressions: {total_individual}")
    print(f"Total Group Expressions: {total_group}")
    print(f"Grand Total: {total_all}")
    
    return all_good, total_individual, total_group, total_all

def validate_category_distribution():
    """Validate Category Distribution table totals."""
    print("\n=== Category Distribution Validation ===")
    
    # Category Distribution data
    categories = [
        {'name': 'Ship', 'individuals': 11461, 'groups': 10402, 'instance_expr': 79251, 'group_expr': 49272},
        {'name': 'Large Vehicle', 'individuals': 17425, 'groups': 18496, 'instance_expr': 121593, 'group_expr': 95356},
        {'name': 'Small Vehicle', 'individuals': 41353, 'groups': 53682, 'instance_expr': 262831, 'group_expr': 282848},
        {'name': 'Building', 'individuals': 10341, 'groups': 3012, 'instance_expr': 66038, 'group_expr': 12048},
        {'name': 'Storage Tank', 'individuals': 2985, 'groups': 3451, 'instance_expr': 19537, 'group_expr': 16071},
        {'name': 'Harbor', 'individuals': 9164, 'groups': 6290, 'instance_expr': 72248, 'group_expr': 28613},
        {'name': 'Swimming Pool', 'individuals': 3147, 'groups': 1999, 'instance_expr': 23355, 'group_expr': 10011},
        {'name': 'Tennis Court', 'individuals': 3492, 'groups': 2364, 'instance_expr': 25116, 'group_expr': 9959},
        {'name': 'Soccer Ball Field', 'individuals': 1781, 'groups': 569, 'instance_expr': 13939, 'group_expr': 2368},
        {'name': 'Roundabout', 'individuals': 924, 'groups': 278, 'instance_expr': 6452, 'group_expr': 1220},
        {'name': 'Basketball Court', 'individuals': 959, 'groups': 636, 'instance_expr': 7339, 'group_expr': 2757},
        {'name': 'Bridge', 'individuals': 3300, 'groups': 1267, 'instance_expr': 23085, 'group_expr': 5269},
        {'name': 'Ground Track Field', 'individuals': 1368, 'groups': 208, 'instance_expr': 9111, 'group_expr': 868},
        {'name': 'Plane', 'individuals': 10774, 'groups': 7260, 'instance_expr': 78808, 'group_expr': 32057},
        {'name': 'Helicopter', 'individuals': 354, 'groups': 266, 'instance_expr': 2636, 'group_expr': 1144},
        {'name': 'Water', 'individuals': 8838, 'groups': 2917, 'instance_expr': 70050, 'group_expr': 11668},
        {'name': 'Road', 'individuals': 0, 'groups': 3018, 'instance_expr': 0, 'group_expr': 12072},
        {'name': 'Agriculture', 'individuals': 0, 'groups': 2342, 'instance_expr': 0, 'group_expr': 9368},
        {'name': 'Barren', 'individuals': 0, 'groups': 1709, 'instance_expr': 0, 'group_expr': 6836},
        {'name': 'Baseball Diamond', 'individuals': 1049, 'groups': 381, 'instance_expr': 7965, 'group_expr': 1576},
        {'name': 'Forest', 'individuals': 0, 'groups': 2850, 'instance_expr': 0, 'group_expr': 11400},
        {'name': 'Vehicle Pair', 'individuals': 0, 'groups': 7597, 'instance_expr': 0, 'group_expr': 30388},
    ]
    
    # Calculate totals
    total_individuals = sum(cat['individuals'] for cat in categories)
    total_groups = sum(cat['groups'] for cat in categories)
    total_instance_expr = sum(cat['instance_expr'] for cat in categories)
    total_group_expr = sum(cat['group_expr'] for cat in categories)
    
    print(f"Total Individual Instances: {total_individuals}")
    print(f"Total Groups: {total_groups}")
    print(f"Total Instance Expressions: {total_instance_expr}")
    print(f"Total Group Expressions: {total_group_expr}")
    
    return total_individuals, total_groups, total_instance_expr, total_group_expr

def validate_llm_enhancement():
    """Validate LLM Enhancement table totals."""
    print("\n=== LLM Enhancement Validation ===")
    
    # LLM Enhancement data
    llm_stats = {
        'Rule-Based Expressions': {'train': 371360, 'val': 134834, 'total': 506194},
        'LLM Enhanced (Language Variations)': {'train': 364396, 'val': 132499, 'total': 496895},
        'LLM Unique (Visual Details)': {'train': 382038, 'val': 137396, 'total': 519434},
        'Total Expressions': {'train': 1117794, 'val': 404729, 'total': 1522523},
    }
    
    # Validate internal consistency
    all_good = True
    for metric, values in llm_stats.items():
        if metric != 'Total Expressions':  # Skip the total row for now
            calculated_total = values['train'] + values['val']
            if calculated_total != values['total']:
                print(f"❌ {metric}: {values['train']} + {values['val']} = {calculated_total}, but total shows {values['total']}")
                all_good = False
            else:
                print(f"✅ {metric}: {values['train']} + {values['val']} = {values['total']}")
    
    # Check if individual totals sum to grand total
    subtotal_train = (llm_stats['Rule-Based Expressions']['train'] + 
                      llm_stats['LLM Enhanced (Language Variations)']['train'] + 
                      llm_stats['LLM Unique (Visual Details)']['train'])
    subtotal_val = (llm_stats['Rule-Based Expressions']['val'] + 
                    llm_stats['LLM Enhanced (Language Variations)']['val'] + 
                    llm_stats['LLM Unique (Visual Details)']['val'])
    subtotal_total = (llm_stats['Rule-Based Expressions']['total'] + 
                      llm_stats['LLM Enhanced (Language Variations)']['total'] + 
                      llm_stats['LLM Unique (Visual Details)']['total'])
    
    print(f"\n--- LLM Enhancement Totals Check ---")
    print(f"Sum of train components: {subtotal_train} vs Total Expressions train: {llm_stats['Total Expressions']['train']}")
    print(f"Sum of val components: {subtotal_val} vs Total Expressions val: {llm_stats['Total Expressions']['val']}")
    print(f"Sum of total components: {subtotal_total} vs Total Expressions total: {llm_stats['Total Expressions']['total']}")
    
    if subtotal_train == llm_stats['Total Expressions']['train']:
        print("✅ Train totals match")
    else:
        print("❌ Train totals don't match")
        all_good = False
        
    if subtotal_val == llm_stats['Total Expressions']['val']:
        print("✅ Val totals match")
    else:
        print("❌ Val totals don't match")
        all_good = False
        
    if subtotal_total == llm_stats['Total Expressions']['total']:
        print("✅ Total totals match")
    else:
        print("❌ Total totals don't match")
        all_good = False
    
    return all_good, llm_stats['Total Expressions']['total']

def cross_validate_tables():
    """Perform cross-table validation."""
    print("\n" + "="*60)
    print("CROSS-TABLE VALIDATION")
    print("="*60)
    
    # Get totals from each table
    _, taxonomy_individual, taxonomy_group, taxonomy_total = validate_expression_taxonomy()
    cat_individuals, cat_groups, cat_instance_expr, cat_group_expr = validate_category_distribution()
    _, llm_total = validate_llm_enhancement()
    
    # Expected values from Dataset Statistics Summary
    expected_individual_objects = 128715
    expected_groups = 130994
    expected_individual_expr = 889354
    expected_group_expr = 633169
    expected_total_samples = 259709
    
    # Rule-based expressions from LLM Enhancement table
    expected_rule_based_total = 506194
    
    print(f"\n--- Cross-Table Validation Results ---")
    
    # Check Individual Objects
    if cat_individuals == expected_individual_objects:
        print(f"✅ Individual Objects: Category table ({cat_individuals}) matches Dataset Stats ({expected_individual_objects})")
    else:
        print(f"❌ Individual Objects: Category table ({cat_individuals}) vs Dataset Stats ({expected_individual_objects})")
    
    # Check Groups
    if cat_groups == expected_groups:
        print(f"✅ Groups: Category table ({cat_groups}) matches Dataset Stats ({expected_groups})")
    else:
        print(f"❌ Groups: Category table ({cat_groups}) vs Dataset Stats ({expected_groups})")
    
    # Check that Category table matches Dataset Stats for all expressions
    if cat_instance_expr == expected_individual_expr:
        print(f"✅ Individual Expressions: Category table ({cat_instance_expr}) matches Dataset Stats ({expected_individual_expr})")
    else:
        print(f"❌ Individual Expressions: Category table ({cat_instance_expr}) vs Dataset Stats ({expected_individual_expr})")
    
    if cat_group_expr == expected_group_expr:
        print(f"✅ Group Expressions: Category table ({cat_group_expr}) matches Dataset Stats ({expected_group_expr})")
    else:
        print(f"❌ Group Expressions: Category table ({cat_group_expr}) vs Dataset Stats ({expected_group_expr})")
    
    # Check that Taxonomy table matches Rule-Based expressions from LLM Enhancement
    print(f"\n--- Expression Taxonomy vs Rule-Based Check ---")
    print(f"⚠️  NOTE: Taxonomy table represents ONLY rule-based expressions, not all expressions")
    if taxonomy_total == expected_rule_based_total:
        print(f"✅ Expression Taxonomy total ({taxonomy_total}) matches Rule-Based Expressions ({expected_rule_based_total})")
    else:
        print(f"❌ Expression Taxonomy total ({taxonomy_total}) vs Rule-Based Expressions ({expected_rule_based_total})")
    
    # Check LLM total vs combined expressions
    combined_expressions = expected_individual_expr + expected_group_expr
    if llm_total == combined_expressions:
        print(f"✅ LLM Enhancement total ({llm_total}) matches combined expressions ({combined_expressions})")
    else:
        print(f"❌ LLM Enhancement total ({llm_total}) vs combined expressions ({combined_expressions})")

def main():
    """Run all validations."""
    print("AERIALSEG DATASET TABLE VALIDATION")
    print("="*50)
    
    # Run individual table validations
    dataset_stats_valid = validate_dataset_statistics()
    
    # Run cross-table validation
    cross_validate_tables()
    
    print(f"\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    if dataset_stats_valid:
        print("✅ Dataset Statistics Summary table is internally consistent")
    else:
        print("❌ Dataset Statistics Summary table has inconsistencies")
    
    print("\n🔍 Review the detailed output above for any cross-table discrepancies.")
    print("📊 All major totals should align between tables for data integrity.")

if __name__ == "__main__":
    main()