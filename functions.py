def create_train_valid_dirs(doggo_types, data_dir):

    """

    :param doggo_types: A list holding all the subdirectories to be made for each breed
    :param data_dir: The directory that the images are held in
    :return: Nothing. Creates train/validation directories for each breed
    """

    import os
    from os.path import join
    import pandas as pd
    from shutil import copyfile

    os.chdir(join(data_dir, 'train'))

    for doggo in doggo_types:

        subset = df_train.loc[df_train['breed'] == doggo, :]

        print('Creating folder for %s' % (doggo))
        print('Number of %s: %s' % (doggo, subset.shape[0]))

        train_files = subset['file'].sample(frac=0.7)
        valid_files = subset['file'][~subset['file'].isin(train_files)]

        train_folder_path = data_dir + '/' + 'train/' + doggo
        valid_folder_path = data_dir + '/' + 'validation/' + doggo

        for f_path in [train_folder_path, valid_folder_path]:

            if not os.path.exists(f_path):
                os.makedirs(f_path)

        print('Moving images to train directory..')

        for i, file in enumerate(train_files):
            copyfile(file, join(train_folder_path, doggo + '_' + str(i) + '.jpg'))

        print('Moving images to validation directory..')

        for i, file in enumerate(valid_files):
            copyfile(file, join(valid_folder_path, doggo + '_' + str(i) + '.jpg'))
