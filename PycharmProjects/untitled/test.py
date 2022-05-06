import streamlit as st
import cv2
from PIL import Image
import numpy as np
import face_recognition as fr
import os
import cv2
import imutils
from skimage.metrics import structural_similarity
from tensorflow import keras
import pytesseract as py
import tensorflow as tf
import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
import time
import matplotlib.pyplot as plt
import seaborn as sns



#read base
path = "inp_img"
images = []
classnames = []
myList = os.listdir(path)
for cl in myList:
    curimg = cv2.imread(f"{path}/{cl}")
    images.append(curimg)
    classnames.append(os.path.splitext(cl)[0])

def findEnc(images):
    encList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encList.append(encode)
    return encList

encListKnown = findEnc(images)

def RealReco():
    cap = cv2.VideoCapture(0)
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    name=""

    facesCurFrame = fr.face_locations(imgS)
    encCurFrame = fr.face_encodings(imgS, facesCurFrame)

    for encface, faceloc in zip(encCurFrame, facesCurFrame):
        matches = fr.compare_faces(encListKnown, encface)
        faceDis = fr.face_distance(encListKnown, encface)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classnames[matchIndex].upper()

        else:
            name = "UNKNOWN"
            faceDistance = ""
        #print(name)
        y1, x2, y2, x1 = faceloc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    return img,name



import sqlite3
conn = sqlite3.connect('data.db')
c = conn.cursor()

def create_chequestable():
	c.execute('CREATE TABLE IF NOT EXISTS cheques(bank TEXT,account TEXT,name TEXT, datee TEXT,ruppees TEXT, pay TEXT,micr TEXT, status TEXT,app TEXT)')

def add_chequesdata(bank,account,name, datee,ruppees, pay,micr, status,app):
	c.execute('INSERT INTO cheques(bank,account,name, datee,ruppees, pay,micr, status,app) VALUES (?,?,?,?,?,?,?,?,?)',(bank,account,name, datee,ruppees, pay,micr, status,app))
	conn.commit()

def cheque_account(account):
	c.execute('SELECT * FROM cheques WHERE account =? ; ',[account])
	data = c.fetchall()
	return data
def all_cheques():
	c.execute('SELECT * FROM cheques  ')
	data = c.fetchall()
	return data


#defining a function to sort contours
def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

# Works well with images of different dimensions
def orb_sim(img1, img2):
    # SIFT is no longer available in cv2 so using ORB
    orb = cv2.ORB_create()

    # detect keypoints and descriptors
    kp_a, desc_a = orb.detectAndCompute(img1, None)
    kp_b, desc_b = orb.detectAndCompute(img2, None)

    # define the bruteforce matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # perform matches.
    matches = bf.match(desc_a, desc_b)
    # Look for similar regions with distance < 50. Goes from 0 to 100 so pick a number between.
    similar_regions = [i for i in matches if i.distance < 20]
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)

def main():

    names = ['Aymen Ouhiba', 'Rebecca Briggs']
    usernames = ['Aymen', 'rbriggs']
    passwords = ['123', '456']
    hashed_passwords = stauth.Hasher(passwords).generate()
    authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
                                        'some_cookie_name', 'some_signature_key', cookie_expiry_days=30)
    btnnn = st.button("Real Time Recognise")

    name, authentication_status, username = authenticator.login('Login', 'main')

    namee=""

    if "run" not in st.session_state:
        st.session_state["run"]=0

    if btnnn  & (st.session_state.run==0):
        realimg, namee = RealReco()

        st.image(realimg)
        st.session_state.run == 1


    if (authentication_status ) :
        authenticator.logout('Logout', 'main')
        st.write('Welcome *%s*' % (name))
        image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        if image_file is not None:
            our_image = Image.open(image_file)
            image1 = np.array(our_image.convert('RGB'))
            resized = cv2.resize(image1, (2300, 1100), interpolation=cv2.INTER_AREA)
            dst = cv2.fastNlMeansDenoisingColored(resized, None, 10, 10, 21, 15)
            img = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(img, (3, 3), 5)
            thresh = cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            temp1 = cv2.imread('C:/Users/ASUS/OneDrive/Bureau/IDRBT_Cheque_Image_Dataset/temp/CANARA.jpeg', 0)
            temp2 = cv2.imread('C:/Users/ASUS/OneDrive/Bureau/IDRBT_Cheque_Image_Dataset/temp/AXIS.jpeg', 0)
            temp3 = cv2.imread('C:/Users/ASUS/OneDrive/Bureau/IDRBT_Cheque_Image_Dataset/temp/SYNDICATE.jpeg', 0)
            temp4 = cv2.imread('C:/Users/ASUS/OneDrive/Bureau/IDRBT_Cheque_Image_Dataset/temp/ICICI.jpeg', 0)
            if (orb_sim(thresh, temp1) > orb_sim(thresh, temp2)) & (
                                orb_sim(thresh, temp1) > orb_sim(thresh, temp3)) & (
                                orb_sim(thresh, temp1) > orb_sim(thresh, temp4)):
                temp = temp1
                bk_NAMEBK = 'CANARA'
            elif (orb_sim(thresh, temp2) > orb_sim(thresh, temp1)) & (
                                orb_sim(thresh, temp2) > orb_sim(thresh, temp3)) & (
                                orb_sim(thresh, temp2) > orb_sim(thresh, temp4)):
                temp = temp2
                bk_NAMEBK = 'AXIS'
            elif (orb_sim(thresh, temp3) > orb_sim(thresh, temp2)) & (
                                orb_sim(thresh, temp3) > orb_sim(thresh, temp1)) & (
                                orb_sim(thresh, temp3) > orb_sim(thresh, temp4)):
                temp = temp3
                bk_NAMEBK = 'SYNDICATE'
            else:
                temp = temp4
                bk_NAMEBK = 'ICICI'
                        # Allignement :
            MAX_FEATURES = 2000
            GOOD_MATCH_PERCENT = 0.3
            orb = cv2.ORB_create(MAX_FEATURES)
            keypoints1, descriptors1 = orb.detectAndCompute(thresh, None)
            keypoints2, descriptors2 = orb.detectAndCompute(temp, None)
            matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
            matches = matcher.match(descriptors1, descriptors2)
            matches = sorted(matches, key=lambda x: x.distance)
            numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
            matches = matches[:numGoodMatches]
            imMatches = cv2.drawMatches(thresh, keypoints1, temp, keypoints2, matches, None)
            points1 = np.zeros((len(matches), 2), dtype=np.float32)
            points2 = np.zeros((len(matches), 2), dtype=np.float32)
            for k, match in enumerate(matches):
                points1[k, :] = keypoints1[match.queryIdx].pt
                points2[k, :] = keypoints2[match.trainIdx].pt
            h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
            height, width = temp.shape
            im1Reg = cv2.warpPerspective(temp, h, (width, height))
            thhh2, im_thhh2 = cv2.threshold(im1Reg, 128, 255, cv2.THRESH_BINARY)
            cleaned_cheque = im_thhh2 - thresh
            cleaned_cheque = cv2.medianBlur(cleaned_cheque, 5)

            PAY = cleaned_cheque[400:600, 1600:2231]
            DATE = cleaned_cheque[10:270, 1700:2300]
            RUPPEES = cleaned_cheque[270:480, 168:1930]
            NAME = cleaned_cheque[183:360, 168:1930]
            SIG = cleaned_cheque[573:974, 1714:2300]
            MICR = cleaned_cheque[919:1241, 198:1840]
            NUM_ACC = cleaned_cheque[490:780, 215:1300]

            st.text("Original Image")

            cnts = cv2.findContours(NAME, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                if 172 < y + h:
                    NAME[y:y + h, x:x + w] = 0

            cnts = cv2.findContours(NUM_ACC, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            rect_areas = []
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                rect_areas.append(w * h)
                avg_area = np.mean(rect_areas)

            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                cnt_area = w * h
                if cnt_area < 0.2 * avg_area:
                    NUM_ACC[y:y + h, x:x + w] = 0

            model_cnn = keras.models.load_model("digits.h5")

            cnts = cv2.findContours(DATE, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])
            (cnts, boundingBoxes) = sort_contours(cnts, method="left-to-right")
            roi = []
            for l in range(len(cnts)):
                roi.append(0)
            rect_areas = []

            k = 0
            n = []
            number = ""
            area = 0
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                TP = w * h
                white = TP - cv2.countNonZero(DATE[y:y + h, x:x + w])
                if (white > TP / 5):
                    if (h > 25):
                        roi[k] = DATE[y:y + h, x:x + w]
                        a = roi[k]
                        roi[k] = cv2.resize(roi[k], (28, 28))
                        roi[k] = roi[k].astype('float32')
                        roi[k] = roi[k].reshape(1, 28, 28, 1)
                        roi[k] /= 255
                        number += str(model_cnn.predict(roi[k]).argmax())
                        k = k + 1
            bk_DATE = number

            cnts = cv2.findContours(PAY, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            rect_areas = []
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                rect_areas.append(w * h)
                avg_area = np.mean(rect_areas)

            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                cnt_area = w * h
                if cnt_area < 0.3 * avg_area:
                    PAY[y:y + h, x:x + w] = 0

            cnts = cv2.findContours(PAY, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])
            roi = []
            for l in range(len(cnts)):
                roi.append(0)
            rect_areas = []

            k = 0
            n = []
            number = ""
            area = 0
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                if (h > 30) & (w > 10):
                    roi[k] = PAY[y:y + h, x:x + w]
                    # print(i)
                    roi[k] = cv2.resize(roi[k], (28, 28))
                    roi[k] = roi[k].astype('float32')
                    roi[k] = roi[k].reshape(1, 28, 28, 1)
                    roi[k] /= 255
                    number += str(model_cnn.predict(roi[k]).argmax())
                    k = k + 1
            bk_PAY = number

            py.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

            str_num_ac = py.image_to_string(NUM_ACC, config="--psm 7 -c tessedit_char_whitelist=0123456789")
            str_num_ac = str_num_ac.rstrip()
            str_nom = py.image_to_string(NAME,
                                                 config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ")
            str_nom = str_nom.rstrip()
            str_micr = py.image_to_string(MICR, lang='mcr')
            str_micr = str_micr.rstrip()
            bk_NUM_ACC = str_num_ac
            bk_NAME = str_nom
            bk_MICR = str_micr

            cnts = cv2.findContours(RUPPEES, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                if (y <= 25) | ((130 < y + h) & (1550 < x + w)):
                    RUPPEES[y:y + h, x:x + w] = 0
            str_montant = py.image_to_string(RUPPEES,
                                                     config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ")
            str_montant = str_montant.rstrip()
            bk_RUPPEES = str_montant

            with open("model.json", "r")as f:
                loaded_json_string = f.read()
            sm = keras.models.model_from_json(loaded_json_string)
            sm.load_weights('weights.h5')

            cv2.imwrite('C:/Users/ASUS/PycharmProjects/untitled/sig.jpeg', SIG)

            sig1 = cv2.imread('C:/Users/ASUS/PycharmProjects/untitled/sig.jpeg')
            sig2 = cv2.imread('C:/Users/ASUS/PycharmProjects/SIG/16.jpeg')

            sig1 = cv2.resize(sig1, (150, 150), interpolation=cv2.INTER_NEAREST)
            sig1 = tf.expand_dims(sig1, axis=0)
            sig2 = cv2.resize(sig2, (150, 150), interpolation=cv2.INTER_NEAREST)
            sig2 = tf.expand_dims(sig2, axis=0)

            st.image(resized)
            bk1 = st.text_input("BANK", value=bk_NAMEBK)
            bk2 = st.text_input("NUM ACCOUNT", value=bk_NUM_ACC)
            bk3 = st.text_input("DATE", value=bk_DATE)
            bk4 = st.text_input("NAME", value=bk_NAME)
            bk5 = st.text_input("RUPPEES", value=bk_RUPPEES)
            bk6 = st.text_input("PAY", value=bk_PAY)
            bk7 = st.text_input("MAGNETIC CODE", value=bk_MICR)

            if (sm.predict([sig2, sig1]) > 0.65):
                bool = "VALID"
            else:
                bool = "INVALID"
            st.text("SIGNATURE   " + bool)

            col1, col2 = st.columns([1, 1])

            with col1:
                if st.button("accept"):
                    create_chequestable()
                    app="accepted"
                    add_chequesdata(bk1, bk2, bk4, bk3, bk5, bk6, bk7, bool,app)
                    st.success("Valid cheque stored")
            with col2:
                if st.button("reject"):
                    create_chequestable()
                    app = "rejected"
                    add_chequesdata(bk1, bk2, bk4, bk3, bk5, bk6, bk7, bool,app)
                    st.error("Invalid cheque stored")

            if st.button("histortique"):
                create_chequestable()
                hist_acc = cheque_account(bk2)
                clean_db = pd.DataFrame(hist_acc,
                                                columns=["BANK", "NUM_ACCOUNT", "NAME", "DATE", "RUPPEES", "PAY", "MICR_CODE",
                                                         "STATUS","app"])
                st.dataframe(clean_db)
            if st.button("statistics"):
                hist_acc = cheque_account(bk2)
                clean_db = pd.DataFrame(hist_acc,
                                        columns=["BANK", "NUM_ACCOUNT", "NAME", "DATE", "RUPPEES", "PAY", "MICR_CODE",
                                                 "STATUS", "app"])

                fig1 = plt.figure(figsize=(10, 4))
                sns.countplot(x="app", data=clean_db)
                plt.title('number of approuved cheques')
                st.pyplot(fig1)
                fig2 = plt.figure(figsize=(10, 4))
                sns.countplot(x="STATUS", data=clean_db)
                plt.title('number of valid signature')
                st.pyplot(fig2)
                clean_db['BRANCH_Code'] = clean_db.apply(lambda row: str(row.MICR_CODE).split()[1][6:9], axis=1)
                fig3 = plt.figure(figsize=(10, 4))
                sns.countplot(x="BRANCH_Code", data=clean_db[clean_db['app'] == "rejected"])
                plt.title("Rejected Cheque's Bank Branch")
                st.pyplot(fig3)
                fig1 = plt.figure(figsize=(10, 4))
                sns.countplot(x="NAME", data=clean_db)
                plt.title("Withdrawer's Name")
                st.pyplot(fig1)

        elif authentication_status == False:
            st.error('Username/password is incorrect')
        elif authentication_status == None:
            st.warning('Please enter your username and password')




if __name__ == '__main__':
    main()


