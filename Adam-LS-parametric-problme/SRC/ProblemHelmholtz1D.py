# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:37:46 2023

@author: jamie.taylor
"""

import tensorflow as tf
import numpy as np


sin=tf.math.sin
cos=tf.math.cos
sqrt=tf.math.sqrt
exp=tf.math.exp

w=0.25**0.5


def indicator(x,a,b):
    div1 = (1+tf.math.sign(x-a))/2
    div2 = (1+(tf.math.sign(b-x)))/2
    return div1*div2

def rhs(x):
    return indicator(x,2/3,1)*(x-2/3)*(x-1)*100

def C1(s1,s2,s3,w):
    return 0.

def C2(s1,s2,s3,w):
    return sqrt(s2)*s3*(sin(w/(3*sqrt(s3)))*w*sqrt(s3) + 6*cos(w/(3*sqrt(s3)))*s3 - 6*s3)/(3*w**4*(((sqrt(s3)*sqrt(s2)*(cos(w/sqrt(s3))*sin((2*w)/(3*sqrt(s3))) - sin(w/sqrt(s3))*cos((2*w)/(3*sqrt(s3))))*cos((2*w)/(3*sqrt(s2))) - sin((2*w)/(3*sqrt(s2)))*s3*(sin(w/sqrt(s3))*sin((2*w)/(3*sqrt(s3))) + cos(w/sqrt(s3))*cos((2*w)/(3*sqrt(s3)))))*cos(w/(3*sqrt(s2))) + sin(w/(3*sqrt(s2)))*(s3*(sin(w/sqrt(s3))*sin((2*w)/(3*sqrt(s3))) + cos(w/sqrt(s3))*cos((2*w)/(3*sqrt(s3))))*cos((2*w)/(3*sqrt(s2))) + sqrt(s3)*sin((2*w)/(3*sqrt(s2)))*sqrt(s2)*(cos(w/sqrt(s3))*sin((2*w)/(3*sqrt(s3))) - sin(w/sqrt(s3))*cos((2*w)/(3*sqrt(s3))))))*sqrt(s1)*cos(w/(3*sqrt(s1))) - sin(w/(3*sqrt(s1)))*((s3*sqrt(s2)*(sin(w/sqrt(s3))*sin((2*w)/(3*sqrt(s3))) + cos(w/sqrt(s3))*cos((2*w)/(3*sqrt(s3))))*cos((2*w)/(3*sqrt(s2))) + sqrt(s3)*sin((2*w)/(3*sqrt(s2)))*s2*(cos(w/sqrt(s3))*sin((2*w)/(3*sqrt(s3))) - sin(w/sqrt(s3))*cos((2*w)/(3*sqrt(s3)))))*cos(w/(3*sqrt(s2))) - (sqrt(s3)*s2*(cos(w/sqrt(s3))*sin((2*w)/(3*sqrt(s3))) - sin(w/sqrt(s3))*cos((2*w)/(3*sqrt(s3))))*cos((2*w)/(3*sqrt(s2))) - sin((2*w)/(3*sqrt(s2)))*s3*sqrt(s2)*(sin(w/sqrt(s3))*sin((2*w)/(3*sqrt(s3))) + cos(w/sqrt(s3))*cos((2*w)/(3*sqrt(s3)))))*sin(w/(3*sqrt(s2))))))

def C3(s1,s2,s3,w):
    return s3*(-sin(w/(3*sqrt(s1)))*cos(w/(3*sqrt(s2)))*sqrt(s2) + cos(w/(3*sqrt(s1)))*sin(w/(3*sqrt(s2)))*sqrt(s1))*((-w*sqrt(s3)*sin(w/sqrt(s3)) - 6*cos(w/sqrt(s3))*s3)*cos((2*w)/(3*sqrt(s3))) + (w*sqrt(s3)*cos(w/sqrt(s3)) - 6*s3*sin(w/sqrt(s3)))*sin((2*w)/(3*sqrt(s3))) + 6*s3)/(3*w**4*((-sqrt(s1)*((sqrt(s3)*sin(w/sqrt(s3))*cos((2*w)/(3*sqrt(s2)))*sqrt(s2) + cos(w/sqrt(s3))*sin((2*w)/(3*sqrt(s2)))*s3)*cos(w/(3*sqrt(s2))) + sin(w/(3*sqrt(s2)))*(sqrt(s3)*sin(w/sqrt(s3))*sin((2*w)/(3*sqrt(s2)))*sqrt(s2) - cos(w/sqrt(s3))*s3*cos((2*w)/(3*sqrt(s2)))))*cos(w/(3*sqrt(s1))) + ((sqrt(s3)*sin(w/sqrt(s3))*sin((2*w)/(3*sqrt(s2)))*s2 - cos(w/sqrt(s3))*s3*sqrt(s2)*cos((2*w)/(3*sqrt(s2))))*cos(w/(3*sqrt(s2))) - sin(w/(3*sqrt(s2)))*(sqrt(s3)*sin(w/sqrt(s3))*cos((2*w)/(3*sqrt(s2)))*s2 + cos(w/sqrt(s3))*sin((2*w)/(3*sqrt(s2)))*s3*sqrt(s2)))*sin(w/(3*sqrt(s1))))*cos((2*w)/(3*sqrt(s3))) + (sqrt(s1)*((sqrt(s3)*cos(w/sqrt(s3))*cos((2*w)/(3*sqrt(s2)))*sqrt(s2) - sin(w/sqrt(s3))*sin((2*w)/(3*sqrt(s2)))*s3)*cos(w/(3*sqrt(s2))) + sin(w/(3*sqrt(s2)))*(sqrt(s3)*cos(w/sqrt(s3))*sin((2*w)/(3*sqrt(s2)))*sqrt(s2) + sin(w/sqrt(s3))*s3*cos((2*w)/(3*sqrt(s2)))))*cos(w/(3*sqrt(s1))) - ((sqrt(s3)*cos(w/sqrt(s3))*sin((2*w)/(3*sqrt(s2)))*s2 + sin(w/sqrt(s3))*s3*sqrt(s2)*cos((2*w)/(3*sqrt(s2))))*cos(w/(3*sqrt(s2))) - sin(w/(3*sqrt(s2)))*(sqrt(s3)*cos(w/sqrt(s3))*cos((2*w)/(3*sqrt(s2)))*s2 - sin(w/sqrt(s3))*sin((2*w)/(3*sqrt(s2)))*s3*sqrt(s2)))*sin(w/(3*sqrt(s1))))*sin((2*w)/(3*sqrt(s3)))))

def C4(s1,s2,s3,w):
    return -((sin(w/(3*sqrt(s1)))*sin(w/(3*sqrt(s2)))*sqrt(s2) + cos(w/(3*sqrt(s1)))*cos(w/(3*sqrt(s2)))*sqrt(s1))*s3*((-w*sqrt(s3)*sin(w/sqrt(s3)) - 6*cos(w/sqrt(s3))*s3)*cos((2*w)/(3*sqrt(s3))) + (w*sqrt(s3)*cos(w/sqrt(s3)) - 6*s3*sin(w/sqrt(s3)))*sin((2*w)/(3*sqrt(s3))) + 6*s3))/(3*w**4*((-sqrt(s1)*((sqrt(s3)*sin(w/sqrt(s3))*cos((2*w)/(3*sqrt(s2)))*sqrt(s2) + cos(w/sqrt(s3))*sin((2*w)/(3*sqrt(s2)))*s3)*cos(w/(3*sqrt(s2))) + sin(w/(3*sqrt(s2)))*(sqrt(s3)*sin(w/sqrt(s3))*sin((2*w)/(3*sqrt(s2)))*sqrt(s2) - cos(w/sqrt(s3))*s3*cos((2*w)/(3*sqrt(s2)))))*cos(w/(3*sqrt(s1))) + ((sqrt(s3)*sin(w/sqrt(s3))*sin((2*w)/(3*sqrt(s2)))*s2 - cos(w/sqrt(s3))*s3*sqrt(s2)*cos((2*w)/(3*sqrt(s2))))*cos(w/(3*sqrt(s2))) - sin(w/(3*sqrt(s2)))*(sqrt(s3)*sin(w/sqrt(s3))*cos((2*w)/(3*sqrt(s2)))*s2 + cos(w/sqrt(s3))*sin((2*w)/(3*sqrt(s2)))*s3*sqrt(s2)))*sin(w/(3*sqrt(s1))))*cos((2*w)/(3*sqrt(s3))) + (sqrt(s1)*((sqrt(s3)*cos(w/sqrt(s3))*cos((2*w)/(3*sqrt(s2)))*sqrt(s2) - sin(w/sqrt(s3))*sin((2*w)/(3*sqrt(s2)))*s3)*cos(w/(3*sqrt(s2))) + sin(w/(3*sqrt(s2)))*(sqrt(s3)*cos(w/sqrt(s3))*sin((2*w)/(3*sqrt(s2)))*sqrt(s2) + sin(w/sqrt(s3))*s3*cos((2*w)/(3*sqrt(s2)))))*cos(w/(3*sqrt(s1))) - ((sqrt(s3)*cos(w/sqrt(s3))*sin((2*w)/(3*sqrt(s2)))*s2 + sin(w/sqrt(s3))*s3*sqrt(s2)*cos((2*w)/(3*sqrt(s2))))*cos(w/(3*sqrt(s2))) - sin(w/(3*sqrt(s2)))*(sqrt(s3)*cos(w/sqrt(s3))*cos((2*w)/(3*sqrt(s2)))*s2 - sin(w/sqrt(s3))*sin((2*w)/(3*sqrt(s2)))*s3*sqrt(s2)))*sin(w/(3*sqrt(s1))))*sin((2*w)/(3*sqrt(s3)))))

def C5(s1,s2,s3,w):
    return -((((-6*sqrt(s3)*sqrt(s2)*(cos(w/sqrt(s3)) - cos((2*w)/(3*sqrt(s3))))*cos((2*w)/(3*sqrt(s2))) + sin((2*w)/(3*sqrt(s2)))*(w*sqrt(s3)*cos(w/sqrt(s3)) + 6*s3*sin((2*w)/(3*sqrt(s3)))))*cos(w/(3*sqrt(s2))) - sin(w/(3*sqrt(s2)))*((w*sqrt(s3)*cos(w/sqrt(s3)) + 6*s3*sin((2*w)/(3*sqrt(s3))))*cos((2*w)/(3*sqrt(s2))) + 6*sqrt(s3)*sin((2*w)/(3*sqrt(s2)))*sqrt(s2)*(cos(w/sqrt(s3)) - cos((2*w)/(3*sqrt(s3))))))*sqrt(s1)*cos(w/(3*sqrt(s1))) + sin(w/(3*sqrt(s1)))*((sqrt(s2)*(w*sqrt(s3)*cos(w/sqrt(s3)) + 6*s3*sin((2*w)/(3*sqrt(s3))))*cos((2*w)/(3*sqrt(s2))) + 6*sqrt(s3)*sin((2*w)/(3*sqrt(s2)))*s2*(cos(w/sqrt(s3)) - cos((2*w)/(3*sqrt(s3)))))*cos(w/(3*sqrt(s2))) + sin(w/(3*sqrt(s2)))*(-6*sqrt(s3)*s2*(cos(w/sqrt(s3)) - cos((2*w)/(3*sqrt(s3))))*cos((2*w)/(3*sqrt(s2))) + sin((2*w)/(3*sqrt(s2)))*sqrt(s2)*(w*sqrt(s3)*cos(w/sqrt(s3)) + 6*s3*sin((2*w)/(3*sqrt(s3)))))))*s3)/(3*w**4*(((sqrt(s3)*sqrt(s2)*(cos(w/sqrt(s3))*sin((2*w)/(3*sqrt(s3))) - sin(w/sqrt(s3))*cos((2*w)/(3*sqrt(s3))))*cos((2*w)/(3*sqrt(s2))) - sin((2*w)/(3*sqrt(s2)))*s3*(sin(w/sqrt(s3))*sin((2*w)/(3*sqrt(s3))) + cos(w/sqrt(s3))*cos((2*w)/(3*sqrt(s3)))))*cos(w/(3*sqrt(s2))) + sin(w/(3*sqrt(s2)))*(s3*(sin(w/sqrt(s3))*sin((2*w)/(3*sqrt(s3))) + cos(w/sqrt(s3))*cos((2*w)/(3*sqrt(s3))))*cos((2*w)/(3*sqrt(s2))) + sqrt(s3)*sin((2*w)/(3*sqrt(s2)))*sqrt(s2)*(cos(w/sqrt(s3))*sin((2*w)/(3*sqrt(s3))) - sin(w/sqrt(s3))*cos((2*w)/(3*sqrt(s3))))))*sqrt(s1)*cos(w/(3*sqrt(s1))) - sin(w/(3*sqrt(s1)))*((s3*sqrt(s2)*(sin(w/sqrt(s3))*sin((2*w)/(3*sqrt(s3))) + cos(w/sqrt(s3))*cos((2*w)/(3*sqrt(s3))))*cos((2*w)/(3*sqrt(s2))) + sqrt(s3)*sin((2*w)/(3*sqrt(s2)))*s2*(cos(w/sqrt(s3))*sin((2*w)/(3*sqrt(s3))) - sin(w/sqrt(s3))*cos((2*w)/(3*sqrt(s3)))))*cos(w/(3*sqrt(s2))) - (sqrt(s3)*s2*(cos(w/sqrt(s3))*sin((2*w)/(3*sqrt(s3))) - sin(w/sqrt(s3))*cos((2*w)/(3*sqrt(s3))))*cos((2*w)/(3*sqrt(s2))) - sin((2*w)/(3*sqrt(s2)))*s3*sqrt(s2)*(sin(w/sqrt(s3))*sin((2*w)/(3*sqrt(s3))) + cos(w/sqrt(s3))*cos((2*w)/(3*sqrt(s3)))))*sin(w/(3*sqrt(s2))))))

def C6(s1,s2,s3,w):
    return s3*(sqrt(s1)*((-6*sqrt(s3)*sqrt(s2)*(sin(w/sqrt(s3)) - sin((2*w)/(3*sqrt(s3))))*cos((2*w)/(3*sqrt(s2))) + sin((2*w)/(3*sqrt(s2)))*(w*sqrt(s3)*sin(w/sqrt(s3)) - 6*s3*cos((2*w)/(3*sqrt(s3)))))*cos(w/(3*sqrt(s2))) - sin(w/(3*sqrt(s2)))*((w*sqrt(s3)*sin(w/sqrt(s3)) - 6*s3*cos((2*w)/(3*sqrt(s3))))*cos((2*w)/(3*sqrt(s2))) + 6*sqrt(s3)*sin((2*w)/(3*sqrt(s2)))*sqrt(s2)*(sin(w/sqrt(s3)) - sin((2*w)/(3*sqrt(s3))))))*cos(w/(3*sqrt(s1))) + ((sqrt(s2)*(w*sqrt(s3)*sin(w/sqrt(s3)) - 6*s3*cos((2*w)/(3*sqrt(s3))))*cos((2*w)/(3*sqrt(s2))) + 6*sqrt(s3)*sin((2*w)/(3*sqrt(s2)))*s2*(sin(w/sqrt(s3)) - sin((2*w)/(3*sqrt(s3)))))*cos(w/(3*sqrt(s2))) + (-6*sqrt(s3)*s2*(sin(w/sqrt(s3)) - sin((2*w)/(3*sqrt(s3))))*cos((2*w)/(3*sqrt(s2))) + sin((2*w)/(3*sqrt(s2)))*sqrt(s2)*(w*sqrt(s3)*sin(w/sqrt(s3)) - 6*s3*cos((2*w)/(3*sqrt(s3)))))*sin(w/(3*sqrt(s2))))*sin(w/(3*sqrt(s1))))/(3*w**4*(((sqrt(s3)*sqrt(s2)*(cos(w/sqrt(s3))*sin((2*w)/(3*sqrt(s3))) - sin(w/sqrt(s3))*cos((2*w)/(3*sqrt(s3))))*cos((2*w)/(3*sqrt(s2))) - sin((2*w)/(3*sqrt(s2)))*s3*(sin(w/sqrt(s3))*sin((2*w)/(3*sqrt(s3))) + cos(w/sqrt(s3))*cos((2*w)/(3*sqrt(s3)))))*cos(w/(3*sqrt(s2))) + sin(w/(3*sqrt(s2)))*(s3*(sin(w/sqrt(s3))*sin((2*w)/(3*sqrt(s3))) + cos(w/sqrt(s3))*cos((2*w)/(3*sqrt(s3))))*cos((2*w)/(3*sqrt(s2))) + sqrt(s3)*sin((2*w)/(3*sqrt(s2)))*sqrt(s2)*(cos(w/sqrt(s3))*sin((2*w)/(3*sqrt(s3))) - sin(w/sqrt(s3))*cos((2*w)/(3*sqrt(s3))))))*sqrt(s1)*cos(w/(3*sqrt(s1))) - sin(w/(3*sqrt(s1)))*((s3*sqrt(s2)*(sin(w/sqrt(s3))*sin((2*w)/(3*sqrt(s3))) + cos(w/sqrt(s3))*cos((2*w)/(3*sqrt(s3))))*cos((2*w)/(3*sqrt(s2))) + sqrt(s3)*sin((2*w)/(3*sqrt(s2)))*s2*(cos(w/sqrt(s3))*sin((2*w)/(3*sqrt(s3))) - sin(w/sqrt(s3))*cos((2*w)/(3*sqrt(s3)))))*cos(w/(3*sqrt(s2))) - (sqrt(s3)*s2*(cos(w/sqrt(s3))*sin((2*w)/(3*sqrt(s3))) - sin(w/sqrt(s3))*cos((2*w)/(3*sqrt(s3))))*cos((2*w)/(3*sqrt(s2))) - sin((2*w)/(3*sqrt(s2)))*s3*sqrt(s2)*(sin(w/sqrt(s3))*sin((2*w)/(3*sqrt(s3))) + cos(w/sqrt(s3))*cos((2*w)/(3*sqrt(s3)))))*sin(w/(3*sqrt(s2))))))

def f_gen_2(x,c5,c6,s3,w):
    return sin(w*x/sqrt(s3))*c5 + cos(w*x/sqrt(s3))*c6 + ((3*x**2 - 5*x + 2)*w**2 - 6*s3)/(3*w**4)

def f_gen_1(x,c1,c2,s,w):
    return c2*sin(w*x/sqrt(s)) + c1*cos(w*x/sqrt(s))

def u_exact(x,s1,s2,s3,w):
    c1=C1(s1,s2,s3,w)
    c2=C2(s1,s2,s3,w)
    c3=C3(s1,s2,s3,w)
    c4=C4(s1,s2,s3,w)
    c5=C5(s1,s2,s3,w)
    c6=C6(s1,s2,s3,w)
    F1 = f_gen_1(x,c1,c2,s1,w)
    F2 = f_gen_1(x,c3,c4,s2,w)
    F3 = f_gen_2(x,c5,c6,s3,w)
    return (indicator(x,0,1/3.)*F1+indicator(x,1/3.,2./3.)*F2+indicator(x,2./3.,1.)*F3)*100