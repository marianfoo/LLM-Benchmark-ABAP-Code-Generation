CLASS z_humaneval_071 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS triangle_area
      IMPORTING
        a             TYPE f
        b             TYPE f
        c             TYPE f
      RETURNING
        VALUE(result) TYPE f.

ENDCLASS.

CLASS z_humaneval_071 IMPLEMENTATION.

  METHOD triangle_area.
    DATA: s    TYPE f,
          area TYPE f.

    IF a + b <= c OR a + c <= b OR b + c <= a.
      result = -1.
      RETURN.
    ENDIF.

    s = ( a + b + c ) / 2.
    area = sqrt( s * ( s - a ) * ( s - b ) * ( s - c ) ).

    result = round( val = area dec = 2 ).
  ENDMETHOD.

ENDCLASS.