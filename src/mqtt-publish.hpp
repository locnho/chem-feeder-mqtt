#ifndef MQTT_HPP
#define MQTT_HPP 1

int mqtt_init();
int mqtt_close();
int mqtt_publish(float ph, bool active, bool alarm);

#endif
