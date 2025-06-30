import xml.etree.ElementTree as ET

def debug_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    print(f"Root tag: {root.tag}")
    
    for obj in root.findall('object'):
        print("\n--- Object ---")
        for child in obj:
            if child.tag == 'expressions':
                print(f"{child.tag}:")
                for expr in child.findall('expression'):
                    print(f"  - {expr.text}")
            else:
                print(f"{child.tag}: {child.text}")

if __name__ == "__main__":
    debug_xml('/tmp/u035679/aerial_seg_clean/aeriald/patches/train/annotations/P2532_patch_020952.xml') 