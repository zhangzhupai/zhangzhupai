use mywork;

# -- 각 나라별 별점 Top3
select *
  from ( select *
    from trip
   where Country = "France"
   order by Rate desc
   limit 3
  ) a
Union
select *
  from ( select *
   from trip
  where Country = "United States"
  order by Rate desc
  limit 3
 ) b
Union
select *
  from ( select *
    from trip
   where Country = "Italy"
   order by Rate desc
   limit 3
  ) c
Union
select *
  from ( select *
    from trip
   where Country = "Spain"
   order by Rate desc
   limit 3
  ) d;

# -- 각 나라별 리뷰수 Top3
select *
  from ( select *
    from trip
   where Country = "France"
   order by Review desc
   limit 3
  ) a
Union
select *
  from ( select *
   from trip
  where Country = "United States"
  order by Review desc
  limit 3
 ) b
Union
select *
  from ( select *
    from trip
   where Country = "Italy"
   order by Review desc
   limit 3
  ) c
Union
select *
  from ( select *
    from trip
   where Country = "Spain"
   order by Review desc
   limit 3
  ) d;

# -- 각 나라별 평균 최저가
select Country, round(avg(Price),2) Avg_Price
  from ( select *
    from trip
   order by Price
  ) a
 group by Country
 order by 2;

# -- 입국 제한이 없는 나라에 속하는 패키지 중 최저가 Top3
select a.Country, a.City, a.Package, a.Price, a.Rate, 
       a.Review, a.Wish, a.Free_Cancell
  from trip a
  join travel_control b
    on a.Country = b.Country
 where b.Travel_Control = "No Restriction"
 order by Price
 limit 3;

# -- 검역이 있는 나라에 속하는 패키지 중 무료취소가 가능한 것
select a.Country, a.City, a.Package, a.Price, a.Rate, a.Review, a.Wish
  from trip a
  join travel_control b
    on a.Country = b.Country
 where b.Travel_Control = "Screening"
   and a.Free_Cancell = 1
 order by Rate desc;
