#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/Joy.h>
#include <unistd.h>

// publisher for a geometry_msgs::Twist topic
ros::Publisher  tf_vel;
geometry_msgs::Twist command_velocity;

void twistCallback(const geometry_msgs::Twist::ConstPtr& twist)
{

  // X vel driven by left joystick forward and aft
  command_velocity.linear.x = twist->linear.x;
  command_velocity.linear.y = twist->linear.y;
  command_velocity.linear.z = twist->linear.z;
 
// heading driven by left joystick left and right
  command_velocity.angular.x = twist->angular.x;
  command_velocity.angular.y = twist->angular.y;
  command_velocity.angular.z = twist->angular.z;

  // publish the cmd vel
  tf_vel.publish(command_velocity);

}


int main(int argc, char** argv)
{

  // init ros
  ros::init(argc, argv, "teleop_joy");

  // create node handle
  ros::NodeHandle node;

  // advertise topic that this node will publish
  tf_vel = node.advertise<geometry_msgs::Twist>("/husky_velocity_controller/cmd_vel", 10);

  // subcscribe to twist topic
  ros::Subscriber twistIn = node.subscribe("/NN_out/cmd_vel", 10, &twistCallback);

  // spin at 50Hz
  ros::Rate rate(50);
  while(ros::ok)
  {
      ros::spinOnce();
      tf_vel.publish(command_velocity);
      rate.sleep();
  }
  return 0;

};


