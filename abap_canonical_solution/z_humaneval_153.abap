CLASS z_humaneval_153 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS strongest_extension
      IMPORTING class_name    TYPE string
                extensions    TYPE string_table
      RETURNING VALUE(result) TYPE string.

ENDCLASS.

CLASS z_humaneval_153 IMPLEMENTATION.

  METHOD strongest_extension.
    DATA: strong       TYPE string,
          my_val       TYPE i,
          val          TYPE i,
          cap_count    TYPE i,
          sm_count     TYPE i,
          char_len     TYPE i,
          idx          TYPE i,
          current_char TYPE c,
          temp_string  TYPE string.

    strong = extensions[ 1 ].

    CLEAR: cap_count, sm_count.
    temp_string = extensions[ 1 ].
    char_len = strlen( temp_string ).
    DO char_len TIMES.
      idx = sy-index - 1.
      current_char = temp_string+idx(1).
      IF current_char >= 'A' AND current_char <= 'Z'.
        cap_count = cap_count + 1.
      ELSEIF current_char >= 'a' AND current_char <= 'z'.
        sm_count = sm_count + 1.
      ENDIF.
    ENDDO.
    my_val = cap_count - sm_count.

    LOOP AT extensions INTO DATA(s).
      CLEAR: cap_count, sm_count.
      char_len = strlen( s ).
      DO char_len TIMES.
        idx = sy-index - 1.
        current_char = s+idx(1).
        IF current_char >= 'A' AND current_char <= 'Z'.
          cap_count = cap_count + 1.
        ELSEIF current_char >= 'a' AND current_char <= 'z'.
          sm_count = sm_count + 1.
        ENDIF.
      ENDDO.
      val = cap_count - sm_count.

      IF val > my_val.
        strong = s.
        my_val = val.
      ENDIF.
    ENDLOOP.

    result = |{ class_name }.{ strong }|.
  ENDMETHOD.

ENDCLASS.