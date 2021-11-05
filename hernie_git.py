#=======================
# area size
#=======================

# compute size of area between rectus left and right
dic = {}
lateral = [2,8,11,14]
patients = [1,2,3,4,7,8,11,13,14,16,17]

for patient in patients:
    if patient not in lateral:

        for i, observation in enumerate(['nativ','valsalva']):

            # path to segmentation
            if patient in patients:
                file = '/media/philipp/HDD/Rohdaten/hernie/labels.hernie/Patient{}_{}.labels_1.tif'.format(str(patient).zfill(3),observation)

            # load segmentation
            if os.path.exists(file):
                a = imread(file)
                zsh, ysh, xsh = a.shape

                # load tomographic data & get pixel spacing
                files = glob.glob('/media/philipp/HDD/Rohdaten/hernie/Dicom_Daten_anonym_zip/Patient{}_{}/*.dcm'.format(str(patient).zfill(3),observation))

                if files:
                    # get pixel spacing
                    file = files[0]
                    image_data, image_header = load(file)
                    y_res, x_res, z_res = image_header.get_voxel_spacing()

                    # get slice thickness
                    ds = pydicom.filereader.dcmread(file)
                    z_res = str(ds[0x180050]).split(' ')[-1].replace('"','')
                    z_res = float(z_res)

                    # compute size of area between rectus left and right
                    area = 0
                    for k in range(zsh):
                        y0,x0 = np.where(a[k]==1)
                        y1,x1 = np.where(a[k]==2)
                        if np.any(x0) and np.any(x1):
                            argmax = np.argmax(x0)
                            argmin = np.argmin(x1)
                            x_max,y_max = x0[argmax],y0[argmax]
                            x_min,y_min = x1[argmin],y1[argmin]
                            area += np.sqrt((x_res*(x_max-x_min))**2 + (y_res*(y_max-y_min))**2) * z_res

                    # compute size of hernia area
                    hernia_area = 0
                    for k in range(zsh):
                        y,x = np.where(a[k]==7)
                        if np.any(x):
                            argmax = np.argmax(x)
                            argmin = np.argmin(x)
                            x_max,x_min = x[argmax],x[argmin]
                            hernia_ += np.sqrt((x_res*(x_max-x_min))**2) * z_res

                    # print and save results
                    print('Patient:', observation+'_'+str(patient).zfill(3), 'Area:', int(round(area)), 'mm^2', 'Hernia area:', int(round(hernia_area)), 'mm^2', 'Resolution:',x_res,y_res,z_res)
                    #dic[observation+'_'+str(patient).zfill(3)] = int(round(area))
                    #np.save('/home/philipp/Seafile/heibox/Projects/Hernien/Data/area_between_rectus.npy',dic)

                else:
                    print('Missing original data:', observation+'_'+str(patient).zfill(3))

#=======================
# git
#=======================

git checkout -b local_dev

git add .
git commit -m ""
git checkout main

git pull git@github.com:biomedisa/hernia-repair.git
git checkout local_dev

git merge main
git push git@github.com:biomedisa/hernia-repair.git local_dev:main
git checkout main

git pull git@github.com:biomedisa/hernia-repair.git
git main -d local_dev

