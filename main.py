from my_functions import *

source = 'test-2.mp4'

save_video = True  # Want to save video? (for video input)
show_video = True  # Set true to display video output
save_img = False  # Set true to save frames as images

# Saving video as output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, frame_size)

cap = cv2.VideoCapture(source)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, frame_size)  # Resize the frame
        original_frame = frame.copy()  # Keep a copy for head cropping
        frame, results = object_detection(frame)

        rider_list = []
        head_list = []
        number_list = []

        for result in results:
            x1, y1, x2, y2, cnf, clas = result
            if clas == 0:  # Rider class
                rider_list.append(result)
            elif clas == 1:  # Head class
                head_list.append(result)
            elif clas == 2:  # Number plate class
                number_list.append(result)

        for rider in rider_list:
            time_stamp = str(time.time())
            x1r, y1r, x2r, y2r, cnfr, clasr = rider

            for head in head_list:
                x1h, y1h, x2h, y2h, cnfh, clash = head
                # Check if head is inside the rider's bounding box
                if inside_box([x1r, y1r, x2r, y2r], [x1h, y1h, x2h, y2h]):
                    try:
                        head_img = original_frame[y1h:y2h, x1h:x2h]
                        helmet_present = img_classify(head_img)
                    except:
                        helmet_present = [None, 0]

                    if helmet_present[0]:  # Helmet detected (True)
                        frame = cv2.rectangle(
                            frame, (x1h, y1h), (x2h, y2h), (0, 255, 0), 2)  # Green box for helmet
                        frame = cv2.putText(
                            frame, 'Helmet', (x1h, y1h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    elif helmet_present[0] is None:  # Poor prediction
                        frame = cv2.rectangle(
                            frame, (x1h, y1h), (x2h, y2h), (255, 255, 0), 2)  # Yellow box for uncertainty
                        frame = cv2.putText(
                            frame, 'Uncertain', (x1h, y1h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    else:  # No helmet detected (False)
                        frame = cv2.rectangle(
                            frame, (x1h, y1h), (x2h, y2h), (0, 0, 255), 2)  # Red box for no helmet
                        frame = cv2.putText(
                            frame, 'No Helmet', (x1h, y1h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        # Save rider's image if helmet is absent
                        try:
                            cv2.imwrite(f'riders_pictures/{time_stamp}.jpg', frame[y1r:y2r, x1r:x2r])
                        except:
                            print('Could not save rider')

                        # Check for number plate within rider's box
                        for num in number_list:
                            x1_num, y1_num, x2_num, y2_num, conf_num, clas_num = num
                            if inside_box([x1r, y1r, x2r, y2r], [x1_num, y1_num, x2_num, y2_num]):
                                try:
                                    num_img = original_frame[y1_num:y2_num, x1_num:x2_num]
                                    cv2.imwrite(f'number_plates/{time_stamp}_{conf_num}.jpg', num_img)
                                except:
                                    print('Could not save number plate')

        if save_video:  # Save the output video
            out.write(frame)
        if save_img:  # Save a single frame as image
            cv2.imwrite('saved_frame.jpg', frame)
        if show_video:  # Show the processed video
            frame = cv2.resize(frame, (900, 450))  # Resize to fit screen
            cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
print('Execution completed')