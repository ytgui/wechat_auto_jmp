from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


def draw():
    glClear(GL_COLOR_BUFFER_BIT)
    glRotatef(0.1, 0, 1, 0)
    glutWireTeapot(0.5)
    glFlush()


glutInit()
glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA)
glutInitWindowSize(400, 400)
glutCreateWindow("First")
glutDisplayFunc(draw)
glutIdleFunc(draw)
glutMainLoop()