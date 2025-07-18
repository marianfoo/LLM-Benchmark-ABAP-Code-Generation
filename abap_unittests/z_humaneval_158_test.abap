CLASS ltcl_test_find_max DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_7 FOR TESTING,
      test_case_8 FOR TESTING,
      test_case_9 FOR TESTING,
      test_case_10 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_find_max IMPLEMENTATION.

  METHOD test_case_1.
    DATA: words TYPE string_table.
    APPEND 'name' TO words.
    APPEND 'of' TO words.
    APPEND 'string' TO words.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_158=>find_max( words )
      exp = 'string'
      msg = 'Test case 1 failed'
    ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: words TYPE string_table.
    APPEND 'name' TO words.
    APPEND 'enam' TO words.
    APPEND 'game' TO words.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_158=>find_max( words )
      exp = 'enam'
      msg = 'Test case 2 failed'
    ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: words TYPE string_table.
    APPEND 'aaaaaaa' TO words.
    APPEND 'bb' TO words.
    APPEND 'cc' TO words.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_158=>find_max( words )
      exp = 'aaaaaaa'
      msg = 'Test case 3 failed'
    ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: words TYPE string_table.
    APPEND 'abc' TO words.
    APPEND 'cba' TO words.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_158=>find_max( words )
      exp = 'abc'
      msg = 'Test case 4 failed'
    ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: words TYPE string_table.
    APPEND 'play' TO words.
    APPEND 'this' TO words.
    APPEND 'game' TO words.
    APPEND 'of' TO words.
    APPEND 'footbott' TO words.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_158=>find_max( words )
      exp = 'footbott'
      msg = 'Test case 5 failed'
    ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: words TYPE string_table.
    APPEND 'we' TO words.
    APPEND 'are' TO words.
    APPEND 'gonna' TO words.
    APPEND 'rock' TO words.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_158=>find_max( words )
      exp = 'gonna'
      msg = 'Test case 6 failed'
    ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: words TYPE string_table.
    APPEND 'we' TO words.
    APPEND 'are' TO words.
    APPEND 'a' TO words.
    APPEND 'mad' TO words.
    APPEND 'nation' TO words.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_158=>find_max( words )
      exp = 'nation'
      msg = 'Test case 7 failed'
    ).
  ENDMETHOD.

  METHOD test_case_8.
    DATA: words TYPE string_table.
    APPEND 'this' TO words.
    APPEND 'is' TO words.
    APPEND 'a' TO words.
    APPEND 'prrk' TO words.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_158=>find_max( words )
      exp = 'this'
      msg = 'Test case 8 failed'
    ).
  ENDMETHOD.

  METHOD test_case_9.
    DATA: words TYPE string_table.
    APPEND 'b' TO words.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_158=>find_max( words )
      exp = 'b'
      msg = 'Test case 9 failed'
    ).
  ENDMETHOD.

  METHOD test_case_10.
    DATA: words TYPE string_table.
    APPEND 'play' TO words.
    APPEND 'play' TO words.
    APPEND 'play' TO words.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_158=>find_max( words )
      exp = 'play'
      msg = 'Test case 10 failed'
    ).
  ENDMETHOD.

ENDCLASS.