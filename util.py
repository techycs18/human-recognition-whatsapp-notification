import os


def list_files(rootPath, validExtensions=(".jpg", ".jpeg", ".png")):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(rootPath):
        # loop over the filenames in the current directory
        for filename in filenames:

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if validExtensions is None or ext.endswith(validExtensions):
                # construct the path to the image and yield it
                image_Path = os.path.join(rootDir, filename)
                yield image_Path
