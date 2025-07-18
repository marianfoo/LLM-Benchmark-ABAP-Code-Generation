CLASS z_humaneval_018 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS how_many_times
      IMPORTING
        VALUE(string)    TYPE string
        VALUE(substring) TYPE string
      RETURNING
        VALUE(count)     TYPE i.
ENDCLASS.

CLASS z_humaneval_018 IMPLEMENTATION.

  METHOD how_many_times.
    DATA: len_string        TYPE i,
          len_substring     TYPE i,
          i                 TYPE i,
          current_substring TYPE string.

    count = 0.
    len_string = strlen( string ).
    len_substring = strlen( substring ).

    IF len_substring = 0 OR len_string = 0.
      RETURN.
    ENDIF.

    i = 0.
    WHILE i <= len_string - len_substring.
      current_substring = substring( val = string off = i len = len_substring ).
      IF current_substring = substring.
        count = count + 1.
      ENDIF.
      i = i + 1.
    ENDWHILE.
  ENDMETHOD.

ENDCLASS.