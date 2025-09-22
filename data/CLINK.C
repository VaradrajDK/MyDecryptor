#include<stdio.h>
#include<malloc.h>
#include<conio.h>
struct cnode
{
int data;
struct cnode *next;
}*p,*q,*start,*i,*y;

int ch,z,a,c,b,t,flag=0,key;

void insertf();
void insertin();
void insertla();
void deletef();
void deletein();
void deletela();

void display()
{
q=start;
while(q->next!=start)
{
 q=q->next;
 }
 y=q;
i=start;
do
{
printf("\t%d",i->data);
i=i->next;
}while(i!=q->next);
}
void main()
{

	char ch1;
	clrscr();
	do
	{
	clrscr();
	printf("\n\tCircular Linked List Menu Card");
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
		p=(struct cnode *)malloc(sizeof(struct cnode));
		printf("\n\tEnter Data Of Node: ");
		scanf("%d",&a);
		p->data=a;
		start=p;
		p->next=start;
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
		i=start;
		do
		{
		printf("\t%d",i->data);
		i=i->next;
		}while(i!=y->next);
		break;
	case 10:
		printf("\n\tSearching");
		printf("\n\tEnter Data Which You Want To Search : ");
		scanf("%d",&key);
		i=start;
		do
		{
			if(i->data==key)
			{
			 flag=1;
			 break;
			 }
			 else
			 {
			  flag=0;
			  i=i->next;
			  }
		}while(i!=y->next);
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
p=(struct cnode*)malloc(sizeof(struct cnode));
printf("\n\tEnter Data Of Node: ");
scanf("%d",&a);
p->data=a;
q=start;
p->next=start;
start=p;
q->next=p;
}


void insertin()
{
p=(struct cnode*)malloc(sizeof(struct cnode));
printf("\n\tEnter Data After Which You Want To Insert Node: ");
scanf("%d",&a);
q=start;
for(i=start;i->next!=start;i=i->next)
{
 if(i->data==a&&i->next!=start)
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
 printf("\n\tElement Which Has Been Entered Does Not Exists or \n\tMaybe Are Trying To Insert Element At Last");
 }
}


void insertla()
{
p=(struct cnode*)malloc(sizeof(struct cnode));

printf("\n\tEnter Data Of Node: ");
scanf("%d",&a);
p->data=a;
p->next=start;
q=start;
while(q->next!=start)
{
 q=q->next;
 }
q->next=p;
}


void deletef()
{
p=start;
q=start;
while(q->next!=start)
{
 q=q->next;
 }
printf("\n\tDeleted Node Which Have Data Is : %d",p->data);
start=p->next;
q->next=start;
free(p);
}


void deletein()
{
p=start;
q=start;
printf("\n\tEnter Data Which You Want To Delete Node: ");
scanf("%d",&b);
for(i=start;i->next!=start;i=i->next)
{
 if(i->data==b&&i->next!=start)
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
printf("\n\tDeleted Node Which Have Data Is : %d",p->data);
q->next=p->next;
free(p);
 }
else
{
 printf("\n\tElement Which Has Been Entered Does Not Exists or\n\t Maybe You Have Enter The Last Element ");
 }
}


void deletela()
{
p=start;
q=start;
while(p->next!=start)
{
 q=p;
 p=p->next;
 }
printf("\n\tDeleted Node Which Have Data Is : %d",p->data);
q->next=start;
free(p);
}