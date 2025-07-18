CLASS ltcl_test_z_humaneval_050 DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.
  PRIVATE SECTION.
    METHODS: test_decode_basic FOR TESTING,
      test_decode_random FOR TESTING,
      test_decode_shift_single FOR TESTING,
      encode_shift IMPORTING s TYPE string
                   RETURNING VALUE(result) TYPE string.
ENDCLASS.

CLASS ltcl_test_z_humaneval_050 IMPLEMENTATION.
  METHOD encode_shift.
    DATA: lv_char        TYPE c LENGTH 1,
          lv_result_char TYPE c LENGTH 1,
          lv_index       TYPE i.

    DO strlen( s ) TIMES.
      lv_index = sy-index - 1.
      lv_char = s+lv_index(1).

      CASE lv_char.
        WHEN 'a'. lv_result_char = 'f'.
        WHEN 'b'. lv_result_char = 'g'.
        WHEN 'c'. lv_result_char = 'h'.
        WHEN 'd'. lv_result_char = 'i'.
        WHEN 'e'. lv_result_char = 'j'.
        WHEN 'f'. lv_result_char = 'k'.
        WHEN 'g'. lv_result_char = 'l'.
        WHEN 'h'. lv_result_char = 'm'.
        WHEN 'i'. lv_result_char = 'n'.
        WHEN 'j'. lv_result_char = 'o'.
        WHEN 'k'. lv_result_char = 'p'.
        WHEN 'l'. lv_result_char = 'q'.
        WHEN 'm'. lv_result_char = 'r'.
        WHEN 'n'. lv_result_char = 's'.
        WHEN 'o'. lv_result_char = 't'.
        WHEN 'p'. lv_result_char = 'u'.
        WHEN 'q'. lv_result_char = 'v'.
        WHEN 'r'. lv_result_char = 'w'.
        WHEN 's'. lv_result_char = 'x'.
        WHEN 't'. lv_result_char = 'y'.
        WHEN 'u'. lv_result_char = 'z'.
        WHEN 'v'. lv_result_char = 'a'.
        WHEN 'w'. lv_result_char = 'b'.
        WHEN 'x'. lv_result_char = 'c'.
        WHEN 'y'. lv_result_char = 'd'.
        WHEN 'z'. lv_result_char = 'e'.
      ENDCASE.

      result = |{ result }{ lv_result_char }|.
    ENDDO.
  ENDMETHOD.

  METHOD test_decode_basic.
    DATA: lv_original TYPE string VALUE 'hello',
          lv_encoded  TYPE string,
          lv_decoded  TYPE string.

    lv_encoded = encode_shift( lv_original ).
    lv_decoded = z_humaneval_050=>decode_shift( lv_encoded ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_decoded
      exp = lv_original ).
  ENDMETHOD.

  METHOD test_decode_random.
    DATA: lv_original TYPE string,
          lv_encoded  TYPE string,
          lv_decoded  TYPE string,
          lv_random   TYPE i,
          lv_length   TYPE i,
          lv_char_num TYPE i,
          lv_char     TYPE c LENGTH 1.

    DO 20 TIMES.
      CLEAR: lv_original, lv_encoded, lv_decoded.

      CALL FUNCTION 'GENERAL_GET_RANDOM_INT'
        EXPORTING
          range  = 11
        IMPORTING
          random = lv_random.
      lv_length = lv_random + 10.

      DO lv_length TIMES.
        CALL FUNCTION 'GENERAL_GET_RANDOM_INT'
          EXPORTING
            range  = 26
          IMPORTING
            random = lv_random.
        lv_char_num = lv_random + 97.

        CASE lv_char_num.
          WHEN 97. lv_char = 'a'.
          WHEN 98. lv_char = 'b'.
          WHEN 99. lv_char = 'c'.
          WHEN 100. lv_char = 'd'.
          WHEN 101. lv_char = 'e'.
          WHEN 102. lv_char = 'f'.
          WHEN 103. lv_char = 'g'.
          WHEN 104. lv_char = 'h'.
          WHEN 105. lv_char = 'i'.
          WHEN 106. lv_char = 'j'.
          WHEN 107. lv_char = 'k'.
          WHEN 108. lv_char = 'l'.
          WHEN 109. lv_char = 'm'.
          WHEN 110. lv_char = 'n'.
          WHEN 111. lv_char = 'o'.
          WHEN 112. lv_char = 'p'.
          WHEN 113. lv_char = 'q'.
          WHEN 114. lv_char = 'r'.
          WHEN 115. lv_char = 's'.
          WHEN 116. lv_char = 't'.
          WHEN 117. lv_char = 'u'.
          WHEN 118. lv_char = 'v'.
          WHEN 119. lv_char = 'w'.
          WHEN 120. lv_char = 'x'.
          WHEN 121. lv_char = 'y'.
          WHEN 122. lv_char = 'z'.
        ENDCASE.

        lv_original = |{ lv_original }{ lv_char }|.
      ENDDO.

      lv_encoded = encode_shift( lv_original ).
      lv_decoded = z_humaneval_050=>decode_shift( lv_encoded ).

      cl_abap_unit_assert=>assert_equals(
        act = lv_decoded
        exp = lv_original ).
    ENDDO.
  ENDMETHOD.

  METHOD test_decode_shift_single.
    DATA: lv_result TYPE string.

    lv_result = z_humaneval_050=>decode_shift( 'f' ).
    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = 'a' ).

    lv_result = z_humaneval_050=>decode_shift( 'e' ).
    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = 'z' ).
  ENDMETHOD.
ENDCLASS.