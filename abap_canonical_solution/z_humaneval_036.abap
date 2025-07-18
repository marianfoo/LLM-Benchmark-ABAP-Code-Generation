CLASS z_humaneval_036 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.
  PUBLIC SECTION.
    CLASS-METHODS fizz_buzz
      IMPORTING
        VALUE(n)      TYPE i
      RETURNING
        VALUE(result) TYPE i.
ENDCLASS.

CLASS z_humaneval_036 IMPLEMENTATION.
  METHOD fizz_buzz.
    DATA: ns      TYPE TABLE OF i,
          s       TYPE string,
          ans     TYPE i,
          i       TYPE i,
          num     TYPE i,
          c       TYPE c LENGTH 1,
          num_str TYPE string,
          idx     TYPE i.

    i = 0.
    WHILE i < n.
      IF i MOD 11 = 0 OR i MOD 13 = 0.
        APPEND i TO ns.
      ENDIF.
      i = i + 1.
    ENDWHILE.

    LOOP AT ns INTO num.
      num_str = num.
      CONCATENATE s num_str INTO s.
    ENDLOOP.

    ans = 0.
    DO strlen( s ) TIMES.
      idx = sy-index - 1.
      c = s+idx(1).
      IF c = '7'.
        ans = ans + 1.
      ENDIF.
    ENDDO.

    result = ans.
  ENDMETHOD.
ENDCLASS.