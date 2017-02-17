import cv2

cam = cv2.VideoCapture(1)
if not cam.isOpened():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise ValueError('Failed to open a capture object.')

i = 0
cont = False

while True:
    _, image = cam.read()

    cv2.imshow("MAIN", image)

    key = cv2.waitKey(1)
    if key == 27:
        break
    if key == ord('c'):
        cont = not cont
        if cont:
            print('[*] Start continious saving')
        else:
            print('[*] Stop continious saving')
    if key == ord('s') or cont:
        filename = 'save/save_{:03d}.png'.format(i)
        i += 1
        print('[+] Saving {}...'.format(filename), end='')
        cv2.imwrite(filename, image)
        print(' done')

cv2.destroyAllWindows()
cam.release()
