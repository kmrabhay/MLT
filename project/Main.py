import cv2
import time
import numpy as np
import os

import DetectChars
import DetectPlates
import PossiblePlate

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False


debug = 0

def main():
    videoFile = "input_video_sample3.mov"
    # videoFrames = readVideo(videoFile)

    # bestString = "something"
    frameCommonString(videoFile)


def frameCommonString(videoFile):
    
    video = cv2.VideoCapture(videoFile);
    fps = video.get(cv2.CAP_PROP_FPS)
    print "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps)
    file = open("result.txt",'w')
    num_frames = 15000;    # no of frames from the video



    ##############  matrix variables ###############
    lastCommonStrings = []
    scoreMatrix = np.zeros((4,5))
    best_sum = -1
    bestString = ""

    # for frame in videoFrames:
    for n in xrange(0, num_frames) :
        ret, frame = video.read()
        # Frame enhancement
        frame = frame[100:-100,1:-1]
        frame[:][:][1] = cv2.equalizeHist(frame[:][:][1])
        frame[:][:][2] = cv2.equalizeHist(frame[:][:][2])
        frame[:][:][3] = cv2.equalizeHist(frame[:][:][3])

        ## call for get the string in the frame
        imgOriginalScene = frame
        plateNo = findPlate(frame)

        if plateNo != None:
            currentString = plateNo.strChars
            lastCommonStrings.append(currentString)
            # print len(lastCommonStrings)


            sum_score = []
            best_sum = -1
            bestString = ""
            if len(lastCommonStrings) >= 5:
                newrow = []
                if debug == 1:
                    print bestString
                    print best_sum
                for string in lastCommonStrings:
                    s = lcs(string,currentString)
                    # print s
                    score = float(len(s))/float(len(string))
                    # print score
                    newrow.append(score)
                    # scoreMatrix[index][4] = score
                scoreMatrix = np.vstack([scoreMatrix, newrow])
                # print scoreMatrix
                for i in range(5):
                    sum_score.append(sum(scoreMatrix[i][:]))
                    # print sum_score[i]
                    # print best_sum
                    if best_sum < sum_score[i]:
                        if lcs(bestString, lastCommonStrings[i]) != lastCommonStrings[i]:
                            best_sum = sum_score[i]
                            bestString = lastCommonStrings[i]

                if debug == 1:
                    print lastCommonStrings[4]
                    print sum_score
                    print scoreMatrix
                confidenceScore = (best_sum/5.0)*100
                if confidenceScore > 80:
                    file.write("Frame number:" + str(n) + " " + bestString+" "+str(confidenceScore)+"\n")
                print "Frame number:" + str(n) + " " +"Best Prediction: ", bestString
                print ("Confidence Score: %.2f" %confidenceScore)
                print "----------------------------"
                ############## Print on vidoe #######################
                if debug ==1:
                    bestStringConfidence = bestString + " C:"+str(int(confidenceScore))+"%"
                    lpWrite(imgOriginalScene, plateNo,bestStringConfidence)           # write license plate text on the image
                    cv2.imshow("imgOriginalScene", imgOriginalScene)                # re-show scene image
                    cv2.imwrite("imgOriginalScene.png", imgOriginalScene)           # write image out to file
                    cv2.waitKey(20)					# hold windows open until user pres
                #####################################################
                if debug == 1:
                    print best_sum
                print "\n"
                lastCommonStrings.remove(lastCommonStrings[0])
                scoreMatrix = np.delete(scoreMatrix, (0), axis=0)
            

    file.close()

    return

#####################################################################################################
def lcs(S,T):
    m = len(S)
    n = len(T)
    counter = [[0]*(n+1) for x in range(m+1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    lcs_set = set()
                    longest = c
                    lcs_set.add(S[i-c+1:i+1])
                elif c == longest:
                    lcs_set.add(S[i-c+1:i+1])

    if len(lcs_set)==0:
        return ""
    else:
        return lcs_set.pop()

###################################################################################################
def findPlate(imgOriginalScene):

    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()         # attempt KNN training

    if blnKNNTrainingSuccessful == False:                               # if KNN training was not successful
        print "\nerror: KNN traning was not successful\n"               # show error message
        return                                                          # and exit program
    # end if


    if imgOriginalScene is None:                            # if image was not read successfully
        print "\nerror: image not read from file \n\n"      # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit program
    # end if

    plateList = DetectPlates.detectPlatesInScene(imgOriginalScene)           # detect plates

    plateList = DetectChars.detectCharsInPlates(plateList)        # detect chars in plates


    if len(plateList) == 0:                          # if no plates were found
        return            # inform user no plates were found
    else:                                                       # else
              
        plateList.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

        plateNo = plateList[0]

        if len(plateNo.strChars) == 0:                     # if no chars were found in the plate
            return                                          # and exit program
        # end if

        stringArray = []
        
        return plateNo
# end main

##################################################################################################
def readVideo(videoFile):
    # Start default camera
    video = cv2.VideoCapture(videoFile);

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # With webcam get(CV_CAP_PROP_FPS) does not work.
    # Let's see for ourselves.

    if int(major_ver)  < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)
    else :
        fps = video.get(cv2.CAP_PROP_FPS)
        print "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps)


    # Number of frames to capture
    num_frames = 120;


    print "Capturing {0} frames".format(num_frames)

    # Start time
    start = time.time()

    # Grab a few frames
    videoFrames = []
    for i in xrange(0, num_frames) :
        ret, frame = video.read()
        frame = frame[100:-100,100:-100]
        frame[:][:][1] = cv2.equalizeHist(frame[:][:][1])
        frame[:][:][2] = cv2.equalizeHist(frame[:][:][2])
        frame[:][:][3] = cv2.equalizeHist(frame[:][:][3])

        

        videoFrames.append(frame)
        
    end = time.time()

    # Time elapsed
    seconds = end - start
    print "Time taken : {0} seconds".format(seconds)

    # Calculate frames per second
    fps  = num_frames / seconds;
    print "Estimated frames per second : {0}".format(fps);

    # Release video
    video.release()
    return videoFrames



###################################################################################################
def lpWrite(imgOriginalScene, plateNo,string):
    textCenterX = 0                             # this will be the center of the area the text will be written to
    textCenterY = 0

    originTextX = 0                          # this will be the bottom left of the area that the text will be written to
    originTextY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = plateNo.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX                      # choose a plain jane font
    fltFontScale = 1 #float(plateHeight) / 30.0                    # base font scale on height of plate area
    intFontThickness = 2 # int(round(fltFontScale * 2.0))           # base font thickness on font scale

    textSize, baseline = cv2.getTextSize(plateNo.strChars, intFontFace, fltFontScale, intFontThickness)        # call getTextSize

            # unpack roatated rect into center point, width and height, and angle
    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = plateNo.rrLocationOfPlateInScene

    # intPlateCenterX = int(intPlateCenterX)              # make sure center is an integer
    # intPlateCenterY = int(intPlateCenterY)

    intPlateCenterX = 100            # make sure center is an integer
    intPlateCenterY = 50

    textCenterX = int(intPlateCenterX)         # the horizontal location of the text area is the same as the plate

    if intPlateCenterY < (sceneHeight * 0.75):                                                  # if the license plate is in the upper 3/4 of the image
        textCenterY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))      # write the chars in below the plate
    else:                                                                                       # else if the license plate is in the lower 1/4 of the image
        textCenterY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))      # write the chars in above the plate
    # end if

    textSizeWidth, textSizeHeight = textSize                # unpack text size width and height

    originTextX = int(textCenterX - (textSizeWidth / 2))           # calculate the lower left origin of the text area
    originTextY = int(textCenterY + (textSizeHeight / 2))          # based on the text area center, width, and height

            # write the text on the image
    cv2.putText(imgOriginalScene, string, (originTextX, originTextY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)
# end function

###################################################################################################
if __name__ == "__main__":
    main()
