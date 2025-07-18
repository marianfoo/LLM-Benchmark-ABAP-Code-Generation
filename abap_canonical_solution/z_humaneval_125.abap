CLASS z_humaneval_125 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.

    CLASS-METHODS: split_words
      IMPORTING txt           TYPE string
      RETURNING VALUE(result) TYPE string_table.

  PRIVATE SECTION.

ENDCLASS.

CLASS z_humaneval_125 IMPLEMENTATION.

  METHOD split_words.
    DATA: lv_char  TYPE c LENGTH 1,
          lv_ascii TYPE i,
          lv_count TYPE i,
          lv_txt   TYPE string,
          lv_index TYPE i.

    lv_txt = txt.

    IF lv_txt = ''.
      result = VALUE #( ).
    ELSEIF lv_txt CA ' '.
      SPLIT lv_txt AT ' ' INTO TABLE result.
    ELSEIF lv_txt CA ','.
      SPLIT lv_txt AT ',' INTO TABLE result.
    ELSE.
      lv_count = 0.
      DO strlen( lv_txt ) TIMES.
        lv_index = sy-index - 1.
        lv_char = lv_txt+lv_index(1).
        IF lv_char >= 'a' AND lv_char <= 'z'.
          CASE lv_char.
            WHEN 'a'. lv_ascii = 0.
            WHEN 'b'. lv_ascii = 1.
            WHEN 'c'. lv_ascii = 2.
            WHEN 'd'. lv_ascii = 3.
            WHEN 'e'. lv_ascii = 4.
            WHEN 'f'. lv_ascii = 5.
            WHEN 'g'. lv_ascii = 6.
            WHEN 'h'. lv_ascii = 7.
            WHEN 'i'. lv_ascii = 8.
            WHEN 'j'. lv_ascii = 9.
            WHEN 'k'. lv_ascii = 10.
            WHEN 'l'. lv_ascii = 11.
            WHEN 'm'. lv_ascii = 12.
            WHEN 'n'. lv_ascii = 13.
            WHEN 'o'. lv_ascii = 14.
            WHEN 'p'. lv_ascii = 15.
            WHEN 'q'. lv_ascii = 16.
            WHEN 'r'. lv_ascii = 17.
            WHEN 's'. lv_ascii = 18.
            WHEN 't'. lv_ascii = 19.
            WHEN 'u'. lv_ascii = 20.
            WHEN 'v'. lv_ascii = 21.
            WHEN 'w'. lv_ascii = 22.
            WHEN 'x'. lv_ascii = 23.
            WHEN 'y'. lv_ascii = 24.
            WHEN 'z'. lv_ascii = 25.
          ENDCASE.

          IF lv_ascii MOD 2 = 1.
            lv_count = lv_count + 1.
          ENDIF.
        ENDIF.
      ENDDO.
      result = VALUE #( ( |{ lv_count }| ) ).
    ENDIF.

  ENDMETHOD.

ENDCLASS.