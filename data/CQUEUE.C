#include<stdio.h>
#include<conio.h>
#define max 20
void main()
{
 int queue[max],front=-1,x,rear=-1,n,i,ch;
 char ch1,ch2,ch3;
 clrscr();
 printf("\n\tEnter Size Of Queue : ");
 scanf("%d",&n);

 do
 {
  clrscr();
  printf("\n\tCircular Queue Menu Card");
  printf("\n1.\tInsert ");
  printf("\n2.\tDelete ");
  printf("\n3.\tDisplay ");
  printf("\n4.\tExit ");
  printf("\n\tEnter Your Choice : ");
  scanf("%d",&ch);
  switch(ch)
  {
   case 1:
	  printf("\n\tInsert ");
	  do
	  {
	   if((front==0&&rear==n-1)||(front==rear+1))
	   {
	    printf("\n\tQueue Is Full !!!!!!");
	    break;
	    }
	   else if(rear==-1)
	   {
	    rear++;
	    front++;
	    }
	   else if(rear==n-1&&front>0)
	   {
	    rear=0;
	    }
	   else
	   {
	    rear++;
	    }
	   printf("\n\tEnter Value To Be Insert : ");
	   scanf("%d",&x);
	   queue[rear]=x;
	   printf("\n\tWould You Like To Insert Again(y/n): ");
	   ch1=getche();
	   }while(ch1=='y');
	  break;
   case 2:
	  printf("\n\tDelete ");
	  do
	  {
	   if(front==-1)
	   {
	    printf("\n\tQueue Is Empty !!!!!");
	    break;
	    }
	   else if(front==n)
	   {
	    front=0;
	    printf("\n\tDeleted Element Is : %d",queue[front]);
	    front++;
	    }
	   else if(front==rear)
	   {
	    printf("\n\tDeleted Element Is : %d",queue[front]);
	    front=-1;
	    rear=-1;
	    }
	   else
	   {
	    printf("\n\tDeleted Element Is : %d",queue[front]);
	    front++;
	    }
	   printf("\n\tWould You Like To Delete Again(y/n): ");
	   ch2=getche();
	   }while(ch2=='y');
	  break;
   case 3:
	  printf("\n\tDisplay ");
	  printf("\n\t");
	  if(front==-1&&rear==-1)
	  {
	   printf("\n\tNo More Element To Display");
	   break;
	   }
	  else if(front>rear)
	  {
	   for(i=front;i<n;i++)
	   {
	    printf("\t%d",queue[i]);
	    }
	   for(i=0;i<=rear;i++)
	   {
	    printf("\t%d",queue[i]);
	    }
	   }
	  else
	  {
	   for(i=front;i<=rear;i++)
	   {
	    printf("\t%d",queue[i]);
	    }
	   }
	  break;
   case 4:
	   exit();

   default:
	  printf("\n\tInvalid Choice ");
	  break;
   }
  printf("\n\tWould You LIke To Choose Another Choice(y/n) : ");
  ch3=getche();
  }while(ch3=='y');

 getch();
 }

