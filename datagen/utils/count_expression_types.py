#!/usr/bin/env python3
import argparse
import os
import xml.etree.ElementTree as ET


THRESHOLD = 1_000_000


def count_expressions(annotations_dir: str) -> tuple[int, int]:
    instance_total = 0
    semantic_total = 0

    for root_dir, _, files in os.walk(annotations_dir):
        for filename in files:
            if not filename.endswith('.xml'):
                continue
            file_path = os.path.join(root_dir, filename)
            try:
                tree = ET.parse(file_path)
            except ET.ParseError:
                continue
            xml_root = tree.getroot()

            for obj in xml_root.findall('object'):
                id_elem = obj.find('id')
                if id_elem is None:
                    continue
                try:
                    target_id = int(id_elem.text)
                except (TypeError, ValueError):
                    continue
                expressions_elem = obj.find('expressions')
                if expressions_elem is None:
                    continue
                count = sum(1 for _ in expressions_elem.findall('expression'))
                if target_id >= THRESHOLD:
                    semantic_total += count
                else:
                    instance_total += count

            groups_elem = xml_root.find('groups')
            if groups_elem is None:
                continue
            for group in groups_elem.findall('group'):
                id_elem = group.find('id')
                if id_elem is None:
                    continue
                try:
                    group_id = int(id_elem.text)
                except (TypeError, ValueError):
                    continue
                expressions_elem = group.find('expressions')
                if expressions_elem is None:
                    continue
                count = sum(1 for _ in expressions_elem.findall('expression'))
                if group_id >= THRESHOLD:
                    semantic_total += count
                else:
                    instance_total += count

    return instance_total, semantic_total


def main() -> None:
    parser = argparse.ArgumentParser(description="Count instance and semantic expressions in AERIAL-D")
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset root (expects split/annotations)')
    args = parser.parse_args()

    splits = [d for d in os.listdir(args.dataset_dir) if os.path.isdir(os.path.join(args.dataset_dir, d))]
    total_instance = 0
    total_semantic = 0

    for split in sorted(splits):
        annotations_dir = os.path.join(args.dataset_dir, split, 'annotations')
        if not os.path.isdir(annotations_dir):
            continue
        instance_count, semantic_count = count_expressions(annotations_dir)
        total_instance += instance_count
        total_semantic += semantic_count
        print(f"Split {split}: instance={instance_count:,} semantic={semantic_count:,}")

    print("\nOverall totals:")
    print(f"Instance expressions: {total_instance:,}")
    print(f"Semantic expressions: {total_semantic:,}")


if __name__ == '__main__':
    main()
