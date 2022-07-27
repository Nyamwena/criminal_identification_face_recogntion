
from django.db import models

import datetime
# Create your models here.
class Criminal_Profile(models.Model):
    alias_name = models.CharField(max_length=30, unique=True)
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
    bio=models.CharField(max_length=200, blank=True)
    crimes_comitted =models.CharField(max_length=200,blank=True, null=True)
    profile_pic = models.ImageField(null=True, blank=True)
    status = models.CharField(max_length=20, blank=True, null=True)
    date = models.DateField(default=datetime.date.today)


    def __str__(self):
        return self.alias_name

class CriminalLog(models.Model):
    #criminal_profile_link = models.ForeignKey(Criminal_Profile,on_delete=models.CASCADE, null=True)
    criminal_profile_link = models.ForeignKey('Criminal_Profile', to_field='alias_name', on_delete=models.CASCADE)
    longitude = models.CharField(max_length=200, blank=True, null=True)
    latitude = models.CharField(max_length=30, blank=True, null=True)
    location = models.CharField(max_length=30, blank=True, null=True)
    date = models.DateTimeField(default=datetime.date.today)


    def __str__(self):
        return self.criminal_profile_link or ''