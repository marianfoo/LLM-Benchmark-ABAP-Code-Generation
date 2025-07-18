CLASS z_humaneval_134 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: check_if_last_char_is_a_letter
      IMPORTING
        txt           TYPE string
      RETURNING
        VALUE(result) TYPE abap_bool.

ENDCLASS.

CLASS z_humaneval_134 IMPLEMENTATION.

  METHOD check_if_last_char_is_a_letter.
    DATA: last_char      TYPE c LENGTH 1,
          last_space_pos TYPE i,
          last_word      TYPE string,
          len            TYPE i,
          pos            TYPE i,
          remaining_len  TYPE i.

    IF txt IS INITIAL.
      result = abap_false.
    ELSE.
      len = strlen( txt ).
      pos = len - 1.
      last_char = txt+pos(1).
      TRANSLATE last_char TO LOWER CASE.


      IF NOT ( last_char >= 'a' AND last_char <= 'z' ).
        result = abap_false.
      ELSE.
        last_space_pos = -1.
        pos = len - 1.
        WHILE pos >= 0.

          IF txt+pos(1) CO ' '.
            last_space_pos = pos.
            EXIT.
          ENDIF.
          pos = pos - 1.
        ENDWHILE.

        IF last_space_pos >= 0.
          pos = last_space_pos + 1.
          remaining_len = len - pos.
          last_word = txt+pos(remaining_len).
        ELSE.
          last_word = txt.
        ENDIF.

        IF strlen( last_word ) = 1.
          result = abap_true.
        ELSE.
          result = abap_false.
        ENDIF.
      ENDIF.
    ENDIF.

  ENDMETHOD.

ENDCLASS.