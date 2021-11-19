import pydicom
import csv
import scipy.ndimage
import numpy as np # linear algebra
import albumentations
from TwoDDATA import correct_dcm
import os
import matplotlib.pyplot as plt
from glob import glob
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
# Some constants
data_path = "../rsna-intracranial-hemorrhage-detection/stage_2_train/"
patients = os.listdir(data_path)
patients.sort()
output_path = working_path = "/output_3d/"
g = glob(data_path + '/*.dcm')
dir_csv = '../rsna-intracranial-hemorrhage-detection/'
train = pd.read_csv(os.path.join(dir_csv, 'stage_2_train.csv'))
test = pd.read_csv(os.path.join(dir_csv, 'stage_2_sample_submission.csv'))

#we need to build the scans, our data is at slice level

"""def link_slices(path):
    scans = {}
    scan_idx = []
    # red every dicom file
    # for each one get the scan id, and dictionary with scan_id(series instance uid) as key  and slices ids as value
    # after that we read the slices for each scan and make the labels (adun toate lables urile?)
    for filename in os.listdir(path):
        slice = pydicom.dcmread(path+filename)
        try:
            scans[slice.SeriesInstanceUID].append(slice.SOPInstanceUID)
        except:
            scans[slice.SeriesInstanceUID] = [slice.SOPInstanceUID]
            scan_idx.append(slice.SeriesInstanceUID)
    return scans, scan_idx

train_scans, train_scan_idx = link_slices('../rsna-intracranial-hemorrhage-detection/stage_2_train/')"""
"""test_scans, test_scan_idx = link_slices('../rsna-intracranial-hemorrhage-detection/stage_2_test/')
with open('scans_reconstruction.csv','w',newline='') as f:  # Python 3
    w = csv.writer(f)
    w.writerow(['list_id','list_items'])
    for key,items in test_scans.items():
        for item in items:
            w.writerow([key,item])

with open('test_scans_reconstruction.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['list_id','list_items'])
    for key,items in test_scans.items():
        for item in items:
            w.writerow([key,item])

"""
with open('scans_reconstruction.csv') as csv_file:
    scans = {}
    scan_idx = []
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
            continue
        else:
            slice = row[1]
            scan = row[0]
            try:
                scans[scan].append(slice)
            except:
                scans[scan] = [slice]
                scan_idx.append(scan)


"""#Loading CT-scans per patient
#To load the full 3D-scan we need to order the single dicom files/slices by the ImagePosition:
zeros = pydicom.dcmread(data_path + 'ID_000012eaf' + '.dcm')
x = np.zeros(zeros.pixel_array.shape)"""

"""zeros.PixelData = x.tobytes()
print(zeros.pixel_array)
filepath = os.path.join(data_path, 'Pad'+'.DCM')
zeros.save_as(filepath)
"""


def load_scan(scan):
    slices =[]
    id = pydicom.dcmread(data_path + scan[0] + '.dcm').SeriesInstanceUID
    for s in scan:
        if s !='Pad':
            dcm = pydicom.dcmread(data_path + s + '.dcm')
            if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
                correct_dcm(dcm)
            slices.append(dcm)
        else:
            continue

    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)


    for s in slices:
        s.SliceThickness = slice_thickness

    """image = np.stack([s.pixel_array for s in slices])

    # return np.array(np.stack[np.zeros((60, 512, 512), dtype=np.int16)])

    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)"""
    return np.array(slices, dtype=np.int16)



def get_pixels_hu(slices):
    try:
        image = np.stack([s.pixel_array for s in slices])

        #return np.array(np.stack[np.zeros((60, 512, 512), dtype=np.int16)])

        # Convert to int16 (from sometimes int16),
        # should be possible as values should always be low enough (<32k)
        image = image.astype(np.int16)

        # Set outside-of-scan pixels to 0
        # The intercept is usually -1024, so air is approximately 0
        image[image == -2000] = 0

        # Convert to Hounsfield units (HU)
        for slice_number in range(len(slices)):

            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope

            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)

            image[slice_number] += np.int16(intercept)

        return np.array(image, dtype=np.int16)
    except:
        return np.zeros((60,512,512), dtype=np.int16)


patient = load_scan(scans[scan_idx[0]])
#imgs = get_pixels_hu(patient)
#print(imgs.shape)

#np.save(output_path + "fullimages_%d.npy" % (id), imgs)


#imgs_to_process = np.load(output_path+'fullimages_{}.npy'.format(id))
#display sample stack
'''

first_patient = load_scan(train_scans[train_scan_idx[0]])
first_patient_pixels = get_pixels_hu(first_patient)
plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()

# Show some slice in the middle
plt.imshow(first_patient_pixels[16], cmap=plt.cm.gray)
plt.show()
'''
"""def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness, scan[0].PixelSpacing[0], scan[0].PixelSpacing[1]], dtype=np.float32)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing"""


"""pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])
print("Shape before resampling\t", first_patient_pixels.shape)
print("Shape after resampling\t", pix_resampled.shape)"""


def plot_3d(image, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces = measure.marching_cubes_classic(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

plot_3d(patient, 400)


MIN_BOUND = -1000.0
MAX_BOUND = 400.0


def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

PIXEL_MEAN = 0.25

def zero_center(image):
    image = image - PIXEL_MEAN
    return image



img = normalize(patient)
img = zero_center(img)

plot_3d(img, 400)
