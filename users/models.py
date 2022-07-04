from operator import mod
from pyexpat import model
from django.db import models
from django.contrib.auth.models import User

import datetime
# Create your models here.


class UserProfile(models.Model):
    user=models.ForeignKey(User,on_delete=models.CASCADE)
    firstname = models.CharField(max_length=30)
    lastname = models.CharField(max_length=30)
    national_id_number = models.CharField(max_length=20, unique=True)
    dob = models.DateField()
    nationality =models.CharField(max_length=30)
    place_of_orgin=models.CharField(max_length=30)
    address=models.CharField(max_length=30)
    phone_number=models.CharField(max_length=30)
    longitude=models.CharField(max_length=200,blank=True, null=True)
    latitude=models.CharField(max_length=30,blank=True, null=True)
    bio=models.CharField(max_length=200)
    crimes_comitted=models.CharField(max_length=200,blank=True, null=True)
    date = models.DateField(default=datetime.date.today)


    def __str__(self): 
        return self.user
