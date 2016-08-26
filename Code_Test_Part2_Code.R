---
title: "BabyName"
author: "Aditya Padala"
date: "12 February 2016"
output: word_document
---

Importing the Text files

```{r}

library(foreign)
library(dplyr)

setwd('/fakepath')

files <-list.files(pattern = ".*.TXT")
data <- data.frame()

for (f in files) {
   data_temp <- read.table(f,header=FALSE,sep=',',col.names = c("State","Sex","Dob","Name","Occ"))
   data <- rbind(data_temp,data)
   data_temp<-data_temp[0,]
} 

```

Question A)

Sol 1:
The text file has five variable: State(2-digit code), Sex (M-Male, F-Female),
Date of birth(Dob) since 1910 to 2014, Name of the baby(2-15 characters), the Number of 
occurences in that specific years (Occ). Files are sorted by state, sex, date of birth,
Occurence . if the occurances are same for multiple names, then it is firther sorted by
names in alphabetical order.

Distortions: The Name has only one name. It doesn't have the middle name and the last name.
It might cause a lot of bias in the dataset. as it doesn't include total set of names and the 
occurances in that year.


Question B)
```{r}
data %>%
  group_by(Sex,Name) %>%
  summarise(
    sum = sum(Occ)
  ) %>%
  arrange(desc(sum)) %>%
  mutate(rank = rank(sum)) %>%
  filter(rank==max(rank))
```
In females "Mary" is the most common name with frequency = 20031 and in males "James" is the most common name with frequency 13139

Question c)
Ambigious Names
```{r}
#most ambigious
year<- function(x) {
  d1<- subset(data, (Dob==x)&(Sex=="M"))
  d2<- subset(data, (Dob==x)&(Sex=="F"))
  d3<- group_by(d1,Sex,Name)%>%
      summarise(sum = sum(Occ))
  d4<- group_by(d2,Sex,Name)%>%
      summarise(sum = sum(Occ))
  d5<-inner_join(d3,d4, by = "Name")
  return (d5[d5$sum.x==d5$sum.y,])
}
year(2013)
year(1945)
```
The most ambigious names in 2013 are Nikita, Arlin, Cree, Devine, Sonam
The most ambigious name in 1945 is Maxie

Question d)
largest increase and decrease
```{r}
percentage<-function(x,y){
  s1<-data%>%
    filter(Dob==x)%>%
    group_by(Name)%>%
    summarise(sum = sum(Occ))
  s2<-data%>%
    filter(Dob==y)%>%
    group_by(Name)%>%
    summarise(sum = sum(Occ))

  s3<-inner_join(s1,s2, by="Name")%>%
    mutate(percent_change = ((sum.y-sum.x)/sum.x)*100)%>%
    arrange(percent_change)
  return (s3[c(nrow(s3),1),]) #Colton
  }

percentage(1980,2014)
```
Largest increase is Colton from 5 occurences in 1980 to 6335 occurences in 2014(126600%)
Largest decrease is Latoya from 2480 occurences in 1980 to 5 occurences in 2014(-99.78%)

Question e)
Yes there might be names that have largest increase and decrease in the whole set.
We can use the percentage function to find out.
Finding out names from 1910 to 2014
```{r}
percentage(1910,2014)
```
largest increase is Joshua from 5 occurences in 1980 to 10764 occurences in 2014(215180%)
largest decrease is Thelma from 2966 occurences in 1980 to 5 occurences in 2014(-99.83%)
These two names are more than previous names in term of percentage changes.
likewise we can also check for other years
```{r}
percentage(1950,2014)
```
largest increase is Ethan from 5 occurences in 1980 to 15619 occurences in 2014(312280%)
largest decrease is Vicki from 5823 occurences in 1980 to 6 occurences in 2014(-99.89%)
These are two such examples that have had an even larger increase or decrease 
in popularity. We can keep checking for more names




Question B)
```{r}
library(ggplot2)
library(gridExtra)
library(reshape)
library(reshape2)
```

checking how names have changed from 1910 to 2014 by gender
```{r}
s_Male<-summarise(group_by(data, Dob,Sex),sum=sum(Occ))%>%filter(Sex=="M")
s_Female<-summarise(group_by(data, Dob,Sex),sum=sum(Occ))%>%filter(Sex=="F")
s<-data%>%
  group_by(Dob,Sex)%>%
  summarise(count = sum(Occ))
s$Dob<-as.factor(s$Dob)
s<-dcast(s,formula = Dob~Sex)
s$percent <- (s$M/s$F)*100

plot_gender<-function(){
  dev.new(width=10, height=6)
  plot(s_Male$Dob, s_Male$sum, col = "Green", type = "l", xlab = "Years", ylab = "Occurance")
  par(new = TRUE)
  plot(s_Female$Dob, s_Female$sum, col = "red", type = "l", axes=F, xlab=NA, ylab=NA)
  par(new = TRUE)
  #ggplot(data = s, aes(x=as.numeric(Dob), y=percent))+geom_line()
  plot(as.numeric(s$Dob), s$percent, col = "blue", type = "l",  axes=F, xlab=NA, ylab=NA)
  axis(side = 4)
  mtext(side = 4, line = 0,"Percentage")
  legend("topleft",
         legend=c("Males", "Females"),
         lty=c(1,1), col=c("Green", "red"))
  par(new = FALSE)
}

plot_gender()
```
By looking at the trend, the green line represents Male Names and the red line represents Female Names
The occurences of male names and female names are almost the same initially,but the trend has changed 
in the last 40-50 years, with male names dominating the females
The blue line shows the ratio of occcurance of male names to female name. it clearly shows that the ratio 
has increased a lot after 1960 which states that males names is equal in female names in occurances.



State-wise "unique" names by gender

```{r}
s1<-data%>%
  group_by(State,Sex)%>%
  summarise( count = length(unique(Name)))

s1$State<-as.character(s1$State)

p1<-ggplot(data=s1, aes(x=State, y=count, fill=Sex))+
  geom_bar(stat="identity",position=position_dodge())+
  theme(legend.position = "bottom") +
  xlab("State") +
  ylab("No of Unique Names") 
  
US_Imm<-read.csv('US_Immigration.csv')
US_Imm$Percent = (US_Imm$Population/sum(US_Imm$Population))*100

p2<-ggplot(data=US_Imm, aes(x=State, y=Percent)) + geom_bar(stat = "identity")+
  theme(axis.text.x = element_text(angle = 90, hjust = 1,vjust = 0.2))+
  xlab("State") +
  ylab("Percentage of Immigrants") 

grid.arrange( p1,p2, ncol=1, nrow =2)
```
Upper plot looks like a lot of uniques names are from Texas,NewYork,Florida,Illinios,California.
This might be because of immigrants in flow into the US

checking the trend with the data collected from https://www.census.gov/hhes/migration/data/acs.html
immigrants per state.

Lower plot tells us that the same states that we observed earlier had large immigrants
flowing into the US. that implies, in CA, IL, TX, NY, FL has a lot of immigrants compared to the other
states. This might be because of the employment opportunity as well.



State-wise Male & Female Names percentage
For plotting the spacial maps for the following i downloaded data of US states and State Code mapping data from  http://statetable.com/  and saved it in "us_code.csv"
```{r}
s2<-data%>%
  group_by(State,Sex)%>%
  summarise( sum = sum(Occ))
s2<-dcast(s2,formula = State~Sex)
s2$Male_Percent = (s2$M/(s2$F+s2$M))
s2$Female_Percent = (s2$F/(s2$F+s2$M))

us_code<- read.csv('US_StateCode.csv')
us_code$region<- tolower(us_code$region)
all_states<-map_data("state")

all_states <- merge(all_states, us_code, by="region", all.x = TRUE)
head(all_states)
Total <- merge(all_states, s2, by = "State")
head(Total)
Total <- Total[Total$region!="district of columbia",]

cnames <- Total%>%
            group_by(State)%>%
            summarise(long=mean(long), lat=mean(lat))

p <- ggplot()
p <- p + 
    geom_polygon(data=Total, aes(x=long, y=lat, group = group, fill=Total$Male_Percent),colour="white") + 
    scale_fill_continuous(low = "thistle2", high = "darkblue", guide="colorbar")
P1 <- p + theme_bw() + labs(fill = "Male Name Relative Percentage" ,title = "Male Name Percentage in the US by State", x="", y="")
P1 + scale_y_continuous(breaks=c()) + scale_x_continuous(breaks=c()) + theme(panel.border =  element_blank())+
   geom_text(data = cnames, aes(long, lat, label = State), size=5, fontface = "bold")
```

Distribution of Popularity of Male Names by State.  The maps shows that there are male names are populated in
states like Nevada, Wyoming

```{r}
q <- ggplot()
q <- q + 
  geom_polygon(data=Total, aes(x=long, y=lat, group = group, fill=Total$Female_Percent),colour="white") + 
  scale_fill_continuous(low = "thistle2", high = "darkred", guide="colorbar")
q1 <- q + theme_bw() + labs(fill = "Female Name Relative Percentage" ,title = "Female Name Percentage in the US by State", x="", y="")
q1 + scale_y_continuous(breaks=c()) + scale_x_continuous(breaks=c()) + theme(panel.border =  element_blank())+  geom_text(data = cnames, aes(long, lat, label = State), size=5, fontface = "bold")

```
Distribution of Popularity of Female Names by State.  The maps shows that there are female names are populated in states on the east coast like New York, Pennsylvania 

