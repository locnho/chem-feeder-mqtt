#include <iostream>
#include <string.h>
#include <mosquitto.h>
#include <stdio.h>

const char *CLIENT_ID = "Chem Feeder";
const char *TOPIC = "chem-feeder/topic";
const char *user = "pi";
const char *password = "Hobby2kho!!!";

void on_connect(struct mosquitto *mosq, void *userdata, int rc)
{
	std::cout << "Connected with result code " << rc << std::endl;
	if (rc == 0) {
		// Successful connection
	} else {
		std::cerr << "Failed to connect: " << mosquitto_connack_string(rc) << std::endl;
	}
}

void on_publish(struct mosquitto *mosq, void *userdata, int mid)
{	
	std::cout << "Message with id " << mid << " published" << std::endl;
}

struct mosquitto *mosq = NULL;

int mqtt_init(void)
{
	mosquitto_lib_init();
	mosq = mosquitto_new(CLIENT_ID, true, NULL);
	if (!mosq) {
		std::cerr << "Out of memory" << std::endl;
		return -1;
	}
	int rc = mosquitto_username_pw_set(mosq, user, password);
        if (rc != MOSQ_ERR_SUCCESS) {
		std::cerr << "Error setting user and password: " << mosquitto_strerror(rc) << std::endl;
		mosquitto_destroy(mosq);
		mosquitto_lib_cleanup();
		return -1;
	}

	mosquitto_connect_callback_set(mosq, on_connect);
	mosquitto_publish_callback_set(mosq, on_publish);

	// Connect to the MQTT broker
	rc = mosquitto_connect(mosq, "localhost", 1883, 60);
	if (rc != MOSQ_ERR_SUCCESS) {
		std::cerr << "Unable to connect: " << mosquitto_strerror(rc) << std::endl;
		mosquitto_destroy(mosq);
		mosquitto_lib_cleanup();
		return -1;
	}
	
	mosquitto_loop_start(mosq);
	return 0;
}

int mqtt_publish(float ph, bool active, bool alarm)
{
	char msg[256];

	sprintf(msg, "%0.1f,%s,%s", ph, active ? "on" : "off", alarm? "alarm":"noalarm");
	int rc = mosquitto_publish(mosq, NULL, TOPIC, strlen(msg), msg, 0, false);
	if (rc != MOSQ_ERR_SUCCESS) {
		std::cerr << "Error publishing: " << msg << "' to topic '" << TOPIC << "'" << std::endl;
	}
	return 0;
}

int mqtt_close(void)
{
	mosquitto_loop_stop(mosq, true);
	mosquitto_lib_cleanup();
	return 0;
}

