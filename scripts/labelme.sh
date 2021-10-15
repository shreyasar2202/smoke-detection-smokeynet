#############################################
# Created by: Anshuman Dewangan
# Date: 2021
#
# Description: Gets LabelMe labelling tool running
#############################################

sudo apt-get update -y &&
sudo DEBIAN_FRONTEND=noninteractive apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" upgrade &&
sudo apt-get install -y \
 apache2 \
 git \
 libapache2-mod-perl2 \
 libcgi-session-perl \
 libapache2-mod-php \
 make \
 php5 \
 libapache2-mod-php5 \
 libcgi-session-perl \
 php &&
sudo a2enmod include &&
sudo a2enmod rewrite &&
sudo a2enmod cgi &&
git clone https://github.com/CSAILVision/LabelMeAnnotationTool.git &&
sudo mv ./LabelMeAnnotationTool/ /var/www/html/LabelMeAnnotationTool/ &&
cd /var/www/html/LabelMeAnnotationTool/ &&
make &&
sudo chown -R www-data:www-data /var/www/html &&
sudo tee /etc/apache2/sites-available/000-default.conf <<EOL
<Directory "/var/www/html/LabelMeAnnotationTool">
   Options Indexes FollowSymLinks MultiViews Includes ExecCGI
   AddHandler cgi-script .cgi
   AllowOverride All
   Require all granted
   AddType text/html .shtml
   AddOutputFilter INCLUDES .shtml
   DirectoryIndex index.shtml
</Directory>
EOL

# sudo vim /etc/apache2/apache2.conf
# ADD TO BOTTOM
#ScriptAlias "/cgi-bin/" "/usr/local/apache2/cgi-bin/"
#<Directory "/var/www/html/LabelMeAnnotationTool/">
#    Options +ExecCGI
#</Directory>
#
#AddHandler cgi-script .cgi .pl
#SetEnv PERL5LIB /var/www/html/LabelMeAnnotationTool/annotationTools/perl
#
#sudo vim /etc/apache2/ports.conf
# *Change LISTEN to 8887*

sudo service apache2 restart

# http://127.0.0.1:8887/LabelMeAnnotationTool/tool.html?mode=f&folder=example_folder&image=1590100955_+01680.jpg