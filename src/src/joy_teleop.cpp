// joy teleop turtlesim example 2015-02-08 LLW
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/Joy.h>
#include <unistd.h>

// publisher for a geometry_msgs::Twist topic
ros::Publisher  tf_vel;
geometry_msgs::Twist command_velocity;

void joyCallback(const sensor_msgs::Joy::ConstPtr& joy)
{
  // declare local variable
  //geometry_msgs::Twist command_velocity;

  // X vel driven by left joystick forward and aft
  command_velocity.linear.x  = 1.0*joy->axes[1];
  command_velocity.linear.y = 0;
  command_velocity.linear.z = 0;
 
// heading driven by left joystick left and right
  command_velocity.angular.x  = 0;
  command_velocity.angular.y = 0;
  command_velocity.angular.z = 0.1*joy->axes[3];

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
  tf_vel =
    node.advertise<geometry_msgs::Twist>("/husky_velocity_controller/cmd_vel", 10);

  // subcscribe to joy topic
  ros::Subscriber sub = node.subscribe("joy", 10, &joyCallback);

  // spin
  
  ros::Rate rate(50);
  while(ros::ok)
  {
      ros::spinOnce();
      tf_vel.publish(command_velocity);
      rate.sleep();
  }

  //ros::spin();

  return 0;

};


