#include<stdio.h>
#include<conio.h>
#define max 20
int queue[max],n,rear=-1,front=-1,x,i;
void display()
{
for(i=front;i<=rear;i++)
{
printf("\t%d",queue[i]);
}
}
void main()
{
int ch;
char ch1,ch2,ch3;
clrscr();
printf("\n\tEnter Size Of Queue : ");
scanf("%d",&n);
do
{

clrscr();
printf("\n\tQueue Menu Card");
printf("\n1.\tInsert");
printf("\n2.\tRemove");
printf("\n3.\tDisplay");
printf("\n4.\tExit");
printf("\n\tEnter Your Choice");
scanf("%d",&ch);
switch(ch)
{
	case 1:
		printf("\n\tInsert");
		do
		{

		if(rear==n-1)
		{
		printf("\n\tQueue Is Full!!!");
		}
		else if(front==-1)
		{
		front=rear=0;
		printf("\n\tEnter Value");
		scanf("\n%d",&x);
		queue[rear]=x;
		}
		else
		{
		rear++;
		printf("\n\tEnter Value");
		scanf("%d",&x);
		queue[rear]=x;

		}
		printf("\n\tYou Want To Insert Another Element(y/n) : ");
		ch1=getch();
		}while(ch1=='y');
		break;

	case 2:
		printf("\n\tRemove");
		do
		{

		if(front==-1||front==rear+1)
		{
		printf("\n\tQueue Is Full!!!!");
		}
		else
		{
		printf("\n\t%d Is Removed",queue[front]);
		front++;

		}
		printf("\n\tYou Want To Removed Another Element(y/n) : ");
		ch2=getch();
		}while(ch2=='y');
		break;

	case 3:
		printf("\n\tDisplay");
		display();
		break;

	case 4:
		exit();

	default:
		printf("\n\tInvalid Choice");
}
printf("\n\tYou Want To Enter Another Choice(y/n) : ");
ch3=getch();
}while(ch3=='y');
getch();
}
