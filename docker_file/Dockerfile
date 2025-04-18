# Docker file for python simple webservice build

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt -y install software-properties-common
RUN apt -y install apache2
RUN apt -y install wget
RUN apt -y install nano


# Python3.10
RUN apt -y install python3
RUN apt -y install python3-pip


# Http settings
ENV APACHE_RUN_USER www-data
ENV APACHE_RUN_GROUP www-data
ENV APACHE_LOG_DIR /var/log/apache2
ENV APACHE_PID_FILE /var/run/apache2.pid
ENV APACHE_RUN_DIR /var/run/apache2
ENV APACHE_LOCK_DIR /var/lock/apache2
RUN mkdir -p $APACHE_RUN_DIR $APACHE_LOCK_DIR $APACHE_LOG_DIR

#RUN mkdir -p /production/www/cgi-bin
#RUN mkdir -p /production/www/lib
#COPY ../cgi-bin /production/www/cgi-bin
#COPY ../lib /production/www/lib
#COPY ../apache2 /etc/apache2
RUN ln -s /etc/apache2/mods-available/cgi.load /etc/apache2/mods-enabled/cgi.load

COPY .. /usr/lib/app
WORKDIR /usr/lib/app

RUN pip3 install -r requirements.txt
RUN chmod a+rwx /usr/lib/cgi-bin
EXPOSE 80
RUN a2enmod headers

ENTRYPOINT [ "/usr/sbin/apache2" ]
CMD ["-D", "FOREGROUND"]
