#include<stdio.h>
#include<malloc.h>
#include<conio.h>
struct node
{
int data;
struct node *next;
}*p,*q,*start,*i;

int ch,z,a,c,b,t,flag=0,key;

void insertf();
void insertin();
void insertla();
void deletef();
void deletein();
void deletela();

void display()
{

for(p=start;p!=NULL;)
{
printf("\t%d",p->data);
p=p->next;
}
}
void main()
{

	char ch1;
	do
	{
	clrscr();
	printf("\n\tLinked List Menu Card");
	printf("\n1.\tCreate");
	printf("\n2.\tInsert First");
	printf("\n3.\tInsert Inbetween");
	printf("\n4.\tInsert Last");
	printf("\n5.\tDelete First");
	printf("\n6.\tDelete Inbetween");
	printf("\n7.\tDelete Last");
	printf("\n8.\tDisplay");
	printf("\n9.\tTraversing");
	printf("\n10.\tSearching");
	printf("\n11.\tExit");
	printf("\n\tEnter Your Choice : ");
	scanf("%d",&ch);
	switch(ch)
	{
	case 1:
		printf("\n\tCreate");
		p=(struct node *)malloc(sizeof(struct node));
		printf("\n\tEnter Data Of Node: ");
		scanf("%d",&a);
		p->data=a;
		p->next=NULL;
		start=p;

		break;
	case 2:
		printf("\n\tInsert First");
		insertf();
		break;

	case 3:
		printf("\n\tInsert Inbetween");
		insertin();
		break;

	case 4:
		printf("\n\tInsert Last");
		insertla();
		break;

	case 5:
		printf("\n\tDelete First");
		deletef();
		break;
	case 6:
		printf("\n\tDelete Inbetween");
		deletein();
		break;
	case 7:
		printf("\n\tDelete Last");
		deletela();
		break;

	case 8:
		printf("\n\tDisplay");
		display();
		break;
	case 9:
		printf("\n\tTraversing");
		printf("\n\tData Inserted In Linked List Is: ");
		printf("\n\tData: ");
		printf("\n");
		for(i=start;i->data!=NULL;i=i->next)
		{
		printf("\t%d",i->data);
		}
		break;
	case 10:
		printf("\n\tSearching");
		printf("\n\tEnter Data Which You Want To Search : ");
		scanf("%d",&key);
		for(i=start;i->data!=NULL;i=i->next)
		{
			if(i->data==key)
			{
			 flag=1;
			 break;
			 }
			 else
			 {
			  flag=0;

			  }
		}
		if(flag==1)
		printf("\n\tThe Searched Element %d Is Found",*i);
		else
		printf("\n\tElement Not Found ");
		break;
	case 11:
		exit();

	default:
		printf("\n\tInvalid Choice");
		break;
	}
 printf("\n\tYou Want To Enter Another Choice(y/n): ");
 ch1=getch();
 }while(ch1=='y');
 getch();
 }


void insertf()
{
p=(struct node*)malloc(sizeof(struct node));
printf("\n\tEnter Data Of Node: ");
scanf("%d",&a);
p->data=a;
p->next=start;
start=p;
}

void insertin()
{
p=(struct node*)malloc(sizeof(struct node));
printf("\n\tEnter Data After Which You Want To Insert Node: ");
scanf("%d",&a);
q=start;
for(i=start;i->next!=NULL;i=i->next)
{
 if(i->data==a)
 {
  flag=1;
  break;
  }
 else
 {
  flag=0;
  }
}
if(flag==1)
{
while(q->data!=a)
{
 q=q->next;
 }
printf("\n\tEnter Data Of Node: ");
scanf("%d",&t);
p->data=t;
p->next=q->next;
q->next=p;
 }
else
{
 printf("\n\tElement Which Has Been Entered Does Not Exists or \n\tMaybe Your Trying To Insert  at Last");
 }
}

void insertla()
{
p=(struct node*)malloc(sizeof(struct node));
printf("\n\tEnter Data Of Node: ");
scanf("%d",&a);
p->data=a;
p->next=NULL;
q=start;
while(q->next!=NULL)
{
q=q->next;
 }
q->next=p;

}

void deletef()
{
p=start;
printf("\n\tDeleted Node Which Have Data Is : %d",p->data);
start=p->next;
free(p);
}

void deletein()
{
printf("\n\tEnter Data Which You Want To Delete : ");
scanf("%d",&b);
p=start;
q=start;
for(i=start;i->data!=NULL;i=i->next)
{
 if(i->data==b&&i->next!=NULL)
 {
  flag=1;
  break;
  }
 else
 {
  flag=0;
  }
 }
if(flag==1)
{
while(p->data!=b)
{
 q=p;
 p=p->next;
 }
 q->next=p->next;
 printf("\n\tDeleted Node Which Have Data Is : %d",p->data);
 free(p);
 }
else
{
 printf("\n\tElement Which Has Been Entered Does Not Exists or \n\tMaybe Your Trying To Delete Last Element");
 }
 }
void deletela()
{
p=start;
q=start;
while(p->next!=NULL)
{
 q=p;
 p=p->next;
 }
printf("\n\tDeleted Node Is : %d",p->data);
q->next=NULL;
free(p);
}