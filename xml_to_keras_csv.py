import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(fullpath, subdir):
    print("looking for files in {}".format(fullpath))
    xml_list = []
    for xml_file in glob.glob(fullpath + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            xmin = member[4][0].text
            xmax = member[4][2].text
            ymin = member[4][1].text
            ymax = member[4][3].text

            width = root.find('size')[0].text
            height = root.find('size')[1].text

            label = member[0].text

            value = ('training_data/images/' + subdir + '/' + root.find('filename').text.lower(),
                     xmin, ymin, xmax, ymax, label
                     )

            xml_list.append(value)
    return xml_list


def main():
    master_list = []
    for subdir in ['cargo', 'vision_target', 'hatch_panel']:
        xml_path = os.path.join(os.getcwd(), 'training_data/annotations/' + subdir)
        new_list = xml_to_csv(xml_path, subdir)
        master_list.extend(new_list)

    column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
    xml_df = pd.DataFrame(master_list, columns=column_name)
    xml_df.to_csv('trisonics_deepspace_training_labels_keras_new.csv', index=None)
    print('Successfully converted xml to csv.')


main()
