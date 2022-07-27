"""security_enforcements_facial_recognition URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views
from recognition import views as recog_views
from users import views as users_views

urlpatterns = [
    path('admin/', admin.site.urls,name='admin-access'),
    path('', recog_views.home, name='home'),
    path('dashboard/', recog_views.dashboard, name='dashboard'),
    path('train/', recog_views.train, name='train'),
    path('add_photos/', recog_views.add_photos, name='add-photos'),
    path('login/',auth_views.LoginView.as_view(template_name='users/login.html'),name='login'),
    path('logout/', auth_views.LogoutView.as_view(template_name='recognition/home.html'), name='logout'),
    path('register/', users_views.register, name='register'),
    path('open_webcam/', recog_views.recognise_the_face, name='recognise_the_face'),
    path('criminal_list/', recog_views.criminals_list, name='criminals_list'),
    path('criminal_profile/<str:pk>/', recog_views.view_criminal_profile, name='criminals_profile'),
    path('update_status/<int:pk>/', recog_views.update_criminal, name='update_status_view'),
    path('update_criminal_record/<int:pk>/', recog_views.update_criminal_record, name='update_criminal_record'),
    path('wanted_criminals/', recog_views.wanted_criminals_report, name='wanted_criminals'),
    path('captured_criminals/', recog_views.captured_criminals_report, name='captured_criminals'),
    path('deseased_criminals/', recog_views.deceased_criminals_report, name='deseased_criminals'),
    path('reports/', recog_views.criminals_report, name='reports'),
    path('cctv_logs/', recog_views.cctv_logs, name='cctv_logs'),
    path('export_to_excel_wanted/', recog_views.export_to_excel, name='export_wanted'),

]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)